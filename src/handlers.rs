use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response, sse::{Event, Sse}},
    Json,
};
use futures::stream::Stream;
use serde_json::json;
use tokio::sync::watch;
use log::error;
use crate::algorithms::{models::{OptimizationProgress, OptimizationRequest, OptimizedCourse, ScheduleChecker, PSO}};

#[derive(Clone)]
pub struct AppState {
    pub status_tx: tokio::sync::broadcast::Sender<OptimizationProgress>,
    pub stop_tx: watch::Sender<bool>,
}

pub async fn stop_handler(
    State(state): State<AppState>,
) -> Result<Response, StatusCode> {
    // Kirim sinyal stop
    if state.stop_tx.send(true).is_err() {
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }
    Ok(Json(json!({ "success": true })).into_response())
}

pub async fn status_handler(
    State(state): State<AppState>,
) -> Sse<impl Stream<Item = Result<Event, axum::Error>> + 'static> {
    let mut rx = state.status_tx.subscribe();
    
    let stream = async_stream::stream! {
        while let Ok(status) = rx.recv().await {
            match serde_json::to_string(&status) {
                Ok(data) => {
                    yield Ok(Event::default().data(data).event("status"));
                }
                Err(e) => error!("Serialization error: {}", e),
            }
        }
    };
    
    Sse::new(stream)
}

pub async fn optimize_handler(
    State(state): State<AppState>,
    Json(req): Json<OptimizationRequest>,
) -> Result<Response, StatusCode> {
    let courses = req.courses.clone();
    let time_preferences = req.time_preferences.clone();
    let parameters = req.parameters.clone();
    let num_runs = 1;

    let status_tx = state.status_tx.clone();
    let stop_rx = state.stop_tx.subscribe();

    if state.stop_tx.send(false).is_err() {
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }

    let mut best_overall_schedule: Option<Vec<OptimizedCourse>> = None;
    let mut best_overall_fitness = f32::INFINITY;
    let mut all_best_fitness = Vec::with_capacity(num_runs);
    
    for i in 0..num_runs {
        let mut pso = PSO::new(
            courses.clone(),
            time_preferences.clone(),
            parameters.clone(),
            Some(status_tx.clone()),
           Some(stop_rx.clone()),
        );

        let (best_position, fitness) =
            pso.optimize(Some((i, num_runs)), &mut all_best_fitness).await;

        let schedule = PSO::position_to_schedule(&best_position, &courses);

        if fitness < best_overall_fitness {
            best_overall_fitness = fitness;
            best_overall_schedule = Some(schedule);
        }
    }

    let conflicts = if let Some(ref schedule) = best_overall_schedule {
        let checker = ScheduleChecker::new(time_preferences.clone());
        checker.evaluate_messages(schedule)
    } else {
        (vec![], vec![]) // fallback kosong jika tidak ada jadwal
    };

    let result = json!({
        "success": true,
        "fitness": best_overall_fitness,
        "all_best_fitness": all_best_fitness,
        "schedule": best_overall_schedule,
        "message": conflicts
    });
    
    let mut response = Json(result).into_response();
    response.headers_mut().insert(
        "content-type",
        "application/json".parse().unwrap()
    );
    
    Ok(response)
}