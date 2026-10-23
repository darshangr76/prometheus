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

    loop {
        let (mut socket, _) = listener.accept().await.unwrap();
        println!("Worker connected!");

        tokio::spawn(async move {
            let task = Task {
                batch_id: 1,
                data: vec![1.0, 2.0, 3.0, 4.0],
            };
            
            let json = serde_json::to_string(&task).unwrap();
            socket.write_all(json.as_bytes()).await.unwrap();
            println!("Sent task to worker.");

            let mut buffer = [0; 1024];
            let n = socket.read(&mut buffer).await.unwrap();
            let response = String::from_utf8_lossy(&buffer[..n]);
            println!("Received gradient from worker: {}", response);
        });
    }
}