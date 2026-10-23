use tokio::net::TcpStream;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
struct Task {
    batch_id: u32,
    data: Vec<f32>,
}

#[tokio::main]
async fn main() {
    let mut stream = TcpStream::connect("127.0.0.1:8080").await.unwrap();
    println!("Connected to Master Node.");

    let mut buffer = [0; 1024];
    let n = stream.read(&mut buffer).await.unwrap();
    let json = String::from_utf8_lossy(&buffer[..n]);
    let task: Task = serde_json::from_str(&json).unwrap();
    
    println!("Received Task: batch_id={}, data={:?}", task.batch_id, task.data);

    let gradient = vec![0.1, 0.2, 0.3, 0.4]; 
    
    let response = serde_json::to_string(&gradient).unwrap();
    stream.write_all(response.as_bytes()).await.unwrap();
    println!("Sent gradient back to Master.");
}