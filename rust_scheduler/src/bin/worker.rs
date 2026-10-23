use tokio::net::TcpStream;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
struct Task {
    batch_id: u32,
    data: Vec<f32>,
}

fn compute_gradient(data: &[f32]) -> Vec<f32> {
    let mut gradient = Vec::new();
    for &val in data {
        gradient.push(2.0 * val);
    }
    gradient
}

#[tokio::main]
async fn main() {
    let mut stream = TcpStream::connect("127.0.0.1:8080").await.unwrap();
    println!("Connected to Master Node.");

    loop {
        let mut buffer = [0; 1024];
        match stream.read(&mut buffer).await {
            Ok(0) => {
                println!("Master disconnected.");
                break;
            }
            Ok(n) => {
                let json = String::from_utf8_lossy(&buffer[..n]);
                let task: Task = serde_json::from_str(&json).unwrap();
                
                println!("Received Task: batch_id={}, data={:?}", task.batch_id, task.data);

                let gradient = compute_gradient(&task.data);
                
                let response = serde_json::to_string(&gradient).unwrap();
                stream.write_all(response.as_bytes()).await.unwrap();
                println!("Sent gradient back: {:?}", gradient);
            }
            Err(e) => {
                eprintln!("Error reading from Master: {}", e);
                break;
            }
        }
    }
}