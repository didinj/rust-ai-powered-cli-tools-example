use clap::{ Parser, Subcommand };
use serde::{ Deserialize, Serialize };
use anyhow::{ Context, Result };
use reqwest::Client;
use serde_json::json;
use std::env;
use colored::*;
use std::io::{ self, Read };
use thiserror::Error;

/// AI-powered CLI tool built in Rust
#[derive(Parser)]
#[command(name = "ai-cli")]
#[command(version = "0.1.0")]
#[command(about = "A small AI-powered CLI tool", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Error, Debug)]
pub enum CliError {
    #[error("API key not set. Please export AI_API_KEY before running.")]
    MissingApiKey,

    #[error("Network request failed: {0}")] NetworkError(String),

    #[error("API returned error: {0}")] ApiError(String),
}

// #[derive(Subcommand)]
// enum Commands {
//     /// Ask a single question and get an AI-generated reply
//     Ask {
//         /// The prompt or question to send
//         #[arg()]
//         prompt: Option<String>,

//         /// Model name (optional, defaults to gpt-4o-mini)
//         #[arg(short, long, default_value = "gpt-4o-mini")]
//         model: String,

//         /// Max tokens (response length)
//         #[arg(short = 'n', long, default_value_t = 150)]
//         max_tokens: u32,

//         /// Temperature (controls randomness)
//         #[arg(short = 'T', long, default_value_t = 0.7)]
//         temperature: f32,
//     },
// }

#[derive(Subcommand)]
enum Commands {
    /// Ask a single question and get an AI-generated reply
    Ask {
        #[arg()]
        prompt: Option<String>,

        #[arg(short, long, default_value = "gpt-4o-mini")]
        model: String,

        #[arg(short = 'n', long, default_value_t = 150)]
        max_tokens: u32,

        #[arg(short = 'T', long, default_value_t = 0.7)]
        temperature: f32,
    },

    /// Summarize text input
    Summarize {
        #[arg()]
        text: Option<String>,
    },

    /// Translate text into another language
    Translate {
        #[arg()]
        text: Option<String>,

        /// Target language (e.g., "fr", "es", "id")
        #[arg(short, long, default_value = "en")]
        to: String,
    },

    /// Start an interactive chat session
    Chat {
        #[arg(short, long, default_value = "gpt-4o-mini")]
        model: String,
    },
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    max_tokens: u32,
    temperature: f32,
}

#[derive(Deserialize, Debug)]
struct ChatChoice {
    message: ChatMessageOwned,
}

#[derive(Deserialize, Debug)]
struct ChatMessageOwned {
    content: String,
}

#[derive(Deserialize, Debug)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: Message,
}

#[derive(Debug, Deserialize)]
struct Message {
    content: String,
}

#[derive(Debug, Deserialize)]
struct ApiResponse {
    choices: Vec<Choice>,
}

async fn ask(
    client: &Client,
    api_key: &str,
    model: &str,
    prompt: &str,
    max_tokens: u32,
    temperature: f32
) -> Result<String> {
    let req = ChatRequest {
        model,
        messages: vec![ChatMessage {
            role: "user",
            content: prompt,
        }],
        max_tokens,
        temperature,
    };

    let url = "https://api.openai.com/v1/chat/completions";

    let res = client
        .post(url)
        .bearer_auth(api_key)
        .json(&req)
        .send().await
        .context("Failed to send request")?;

    if !res.status().is_success() {
        let status = res.status();
        let body = res.text().await.unwrap_or_default();
        anyhow::bail!("API error: {} - {}", status, body);
    }

    let completion: ChatResponse = res.json().await.context("Failed to parse response")?;

    let reply = completion.choices
        .get(0)
        .map(|c| c.message.content.clone())
        .unwrap_or_else(|| "No reply found".to_string());

    Ok(reply.trim().to_string())
}

fn get_input_or_stdin(opt: &Option<String>, prompt: &str) -> anyhow::Result<String> {
    if let Some(text) = opt {
        Ok(text.clone())
    } else {
        println!("{}", prompt.blue());
        let mut buffer = String::new();
        std::io::stdin().read_to_string(&mut buffer)?;
        Ok(buffer)
    }
}

async fn send_ai_request(user_input: &str, task: &str, model: &str) -> anyhow::Result<()> {
    let api_key = std::env::var("AI_API_KEY").map_err(|_| CliError::MissingApiKey)?;

    log::debug!("Sending request with model {}", model);

    let client = reqwest::Client::new();
    let full_prompt = format!("{}\n\n{}", task, user_input);

    let payload =
        serde_json::json!({
        "model": model,
        "messages": [
            { "role": "user", "content": full_prompt }
        ],
        "max_tokens": 200,
        "temperature": 0.7
    });

    let res = client
        .post("https://api.openai.com/v1/chat/completions")
        .bearer_auth(&api_key)
        .json(&payload)
        .send().await
        .map_err(|e| CliError::NetworkError(e.to_string()))?;

    let status = res.status();
    if !status.is_success() {
        let err_text = res.text().await.unwrap_or_default();
        return Err(CliError::ApiError(format!("{} - {}", status, err_text)).into());
    }

    let api_response: ApiResponse = res.json().await?;

    if let Some(choice) = api_response.choices.first() {
        println!("{}", "================ AI Response ================".green().bold());
        println!("{}", choice.message.content.white());
        println!("{}", "============================================".green().bold());
    } else {
        println!("{}", "⚠️ No response received from AI.".yellow());
    }

    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    let cli = Cli::parse();

    // match &cli.command {
    //     Commands::Ask { prompt, model, max_tokens, temperature } => {
    //         // Get prompt either from arg or stdin
    //         let user_prompt = if let Some(p) = prompt {
    //             p.clone()
    //         } else {
    //             println!("{}", "Enter your prompt (Ctrl+D to finish):".blue());
    //             let mut buffer = String::new();
    //             io::stdin().read_to_string(&mut buffer)?;
    //             buffer
    //         };

    //         println!("Prompt: {}", user_prompt);
    //         println!("Model: {}", model);
    //         println!("Max tokens: {}", max_tokens);
    //         println!("Temperature: {}", temperature);
    //     }
    // }
    match &cli.command {
        Commands::Summarize { text } => {
            let input_text = get_input_or_stdin(
                text,
                "Paste text to summarize (Ctrl+D to finish):"
            )?;
            send_ai_request(
                &input_text,
                "Summarize the following text briefly:",
                "gpt-4o-mini"
            ).await?;
        }

        Commands::Translate { text, to } => {
            let input_text = get_input_or_stdin(
                text,
                "Paste text to translate (Ctrl+D to finish):"
            )?;
            let prompt = format!("Translate the following text into {}:", to);
            send_ai_request(&input_text, &prompt, "gpt-4o-mini").await?;
        }

        Commands::Chat { model } => {
            println!("{}", "Starting interactive chat (type 'exit' to quit)".cyan().bold());

            let mut history = vec![];

            loop {
                print!("{}", "You: ".blue().bold());
                use std::io::Write;
                std::io::stdout().flush()?;

                let mut input = String::new();
                std::io::stdin().read_line(&mut input)?;
                let input = input.trim();

                if input.eq_ignore_ascii_case("exit") {
                    break;
                }

                history.push(json!({ "role": "user", "content": input }));

                let payload =
                    json!({
            "model": model,
            "messages": history,
            "max_tokens": 200,
            "temperature": 0.7
        });

                let api_key = std::env::var("AI_API_KEY").expect("AI_API_KEY not set");
                let client = reqwest::Client::new();
                let res = client
                    .post("https://api.openai.com/v1/chat/completions")
                    .bearer_auth(api_key)
                    .json(&payload)
                    .send().await?;

                let api_response: ApiResponse = res.json().await?;
                if let Some(choice) = api_response.choices.first() {
                    println!("{}", format!("AI: {}", choice.message.content).green());
                    history.push(
                        json!({
                "role": "assistant",
                "content": choice.message.content
            })
                    );
                }
            }
        }

        _ => {}
    }

    Ok(())
}

// #[tokio::main]
// async fn main() -> Result<()> {
//     dotenvy::dotenv().ok();
//     let cli = Cli::parse();

//     let api_key = env
//         ::var("AI_API_KEY")
//         .context("Please set AI_API_KEY in your environment or .env file")?;

//     let client = Client::new();

//     match &cli.command {
//         Commands::Ask { prompt, model, max_tokens, temperature } => {
//             match ask(&client, &api_key, model, prompt, *max_tokens, *temperature).await {
//                 Ok(reply) => println!("\nAI reply:\n{}\n", reply),
//                 Err(e) => eprintln!("Error: {:?}", e),
//             }
//         }
//     }

//     Ok(())
// }

// #[tokio::main]
// async fn main() -> anyhow::Result<()> {
//     let cli = Cli::parse();

//     match &cli.command {
//         Commands::Ask { prompt, model, max_tokens, temperature } => {
//             // Load API key
//             let api_key = env::var("AI_API_KEY").expect("AI_API_KEY environment variable not set");

//             // Prepare request payload
//             let payload =
//                 json!({
//                 "model": model,
//                 "messages": [
//                     { "role": "user", "content": prompt }
//                 ],
//                 "max_tokens": max_tokens,
//                 "temperature": temperature
//             });

//             // Send request
//             let client = Client::new();
//             let res = client
//                 .post("https://api.openai.com/v1/chat/completions")
//                 .bearer_auth(api_key)
//                 .json(&payload)
//                 .send().await?;

//             if !res.status().is_success() {
//                 let err_text = res.text().await?;
//                 anyhow::bail!("API error: {} - {}", res.status(), err_text);
//             }

//             // Parse response
//             let api_response: ApiResponse = res.json().await?;

//             if let Some(choice) = api_response.choices.first() {
//                 println!("\n🤖 AI Response:\n{}\n", choice.message.content);
//             } else {
//                 println!("No response received from AI.");
//             }
//         }
//     }

//     Ok(())
// }
