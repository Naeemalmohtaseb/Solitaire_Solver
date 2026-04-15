//! Minimal CLI for workspace diagnostics.

use clap::{Parser, Subcommand};

/// Command-line entry point for the Solitaire solver backend.
#[derive(Debug, Parser)]
#[command(name = "solitaire-cli")]
#[command(about = "Draw-3 Klondike solver backend CLI")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// Supported diagnostic commands.
#[derive(Debug, Subcommand)]
enum Command {
    /// Print basic environment and crate health information.
    Doctor,
    /// Print the solver crate version.
    Version,
    /// Print the high-level architecture summary.
    PrintArchitecture,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::Doctor => {
            println!("solitaire-cli {}", solver_core::VERSION);
            println!("solver_core: available");
            println!("architecture: {}", solver_core::architecture_summary());
        }
        Command::Version => {
            println!("{}", solver_core::VERSION);
        }
        Command::PrintArchitecture => {
            println!("{}", solver_core::architecture_summary());
        }
    }
}
