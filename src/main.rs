mod models;
mod pso;
mod algorithms;
mod handlers;

use axum::{
    http::{header, Method, HeaderValue},
    routing::{get, post},
    Router,
};
use std::time::Duration;
use tower_http::cors::CorsLayer;
use tokio::sync::{broadcast, watch};
use handlers::{AppState, optimize_handler, status_handler, stop_handler};

#[tokio::main]
async fn main() {
    env_logger::init();
    
    let (status_tx, _) = broadcast::channel(1024);
    let (stop_tx, stop_rx) = watch::channel(false);
    let state = AppState { status_tx, stop_tx };
    
    // Configure CORS
    let cors = CorsLayer::new()
        .allow_origin("http://localhost:3000".parse::<HeaderValue>().unwrap())
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers([header::CONTENT_TYPE, header::ACCEPT])
        .expose_headers([header::CONTENT_TYPE])
        .allow_credentials(true)
        .max_age(Duration::from_secs(3600));
    
    // Setup routes
    let app = Router::new()
        .route("/optimize", post(optimize_handler))
        .route("/status", get(status_handler))
        .route("/stop", post(stop_handler))
        .layer(cors)
        .with_state(state);
    
    // Start server
    let listener = tokio::net::TcpListener::bind("127.0.0.1:8080").await.unwrap();
    println!("Server running on http://127.0.0.1:8080");
    axum::serve(listener, app).await.unwrap();
}