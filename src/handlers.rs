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
use crate::models::{OptimizationRequest, OptimizationStatus};
use crate::pso::optimizer::PSO;

#[derive(Clone)]
pub struct AppState {
    pub status_tx: tokio::sync::broadcast::Sender<OptimizationStatus>,
    pub stop_tx: watch::Sender<bool>,
}

pub async fn stop_handler(
    State(state): State<AppState>,
) -> Result<Response, StatusCode> {
    // Kirim sinyal stop
    if state.stop_tx.send(true).is_err() {
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }
    // Reset stop_tx ke false setelah mengirim sinyal stop
    if state.stop_tx.send(false).is_err() {
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
    let sum_ruangan = req.sum_ruangan;
    let parameters = req.parameters.clone();
    let status_tx = state.status_tx.clone();
    let stop_rx = state.stop_tx.subscribe(); // Dapatkan receiver untuk sinyal stop
    
    // Jalankan optimisasi di thread terpisah
    let result = tokio::task::spawn(async move {
        let mut pso = PSO::new(
            courses.clone(),
            time_preferences,
            parameters.clone(),
            sum_ruangan
        );
        
        // Jalankan optimisasi secara asinkron
        let best_position = pso.optimize(status_tx, stop_rx).await;
        
        // Evaluasi hasil terbaik
        let (fitness, conflicts) = pso.evaluate_best_position();
        
        // Convert hasil ke jadwal yang dapat digunakan
        let optimized_schedule = PSO::position_to_schedule(&best_position, &courses, sum_ruangan);
        
        serde_json::json!({
            "success": true,
            "fitness": fitness,
            "conflicts": conflicts,
            "schedule": optimized_schedule
        })
    })
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    let mut response = Json(result).into_response();
    response.headers_mut().insert(
        "content-type",
        "application/json".parse().unwrap()
    );
    
    Ok(response)
}