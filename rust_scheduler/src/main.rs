use tokio::net::TcpListener;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
struct Task {
    batch_id: u32,
    data: Vec<f32>,
}

#[tokio::main]
async fn main() {
    let listener = TcpListener::bind("127.0.0.1:8080").await.unwrap();
    println!("Rust Master Node running on 127.0.0.1:8080");

    let (mut socket, _) = listener.accept().await.unwrap();
    println!("Worker connected!");

    let mut total_gradient = vec![0.0; 4];

    for batch_id in 1..=5 {
        let task = Task {
            batch_id,
            data: vec![batch_id as f32, 2.0, 3.0, 4.0],
        };
        
        let json = serde_json::to_string(&task).unwrap();
        socket.write_all(json.as_bytes()).await.unwrap();
        println!("Sent Task: batch_id={}", batch_id);

        let mut buffer = [0; 1024];
        let n = socket.read(&mut buffer).await.unwrap();
        let response = String::from_utf8_lossy(&buffer[..n]);
        let gradient: Vec<f32> = serde_json::from_str(&response).unwrap();
        
        println!("Batch {} | Received gradient: {:?}", batch_id, gradient);
        
        for i in 0..4 {
            total_gradient[i] += gradient[i];
        }
    }

    println!("--- Training Complete ---");
    println!("Aggregated Gradient: {:?}", total_gradient);
}