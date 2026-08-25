#!/usr/bin/env python3
"""
AudioForge CLI - Forensic audio separation and analysis toolkit
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

try:
    from core.analyzer import AudioAnalyzer
    from core.separator import SourceSeparator
    from core.amplifier import StemAmplifier
    from utils.logger import get_logger
except ImportError:
    from audioforge.core.analyzer import AudioAnalyzer
    from audioforge.core.separator import SourceSeparator
    from audioforge.core.amplifier import StemAmplifier
    from audioforge.utils.logger import get_logger

console = Console()
logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        prog="audioforge",
        description="🔧 AudioForge - Forensic audio separation and analysis toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  audioforge analyze recording.wav
  audioforge extract recording.wav --query "helicopter rotor sound"
  audioforge extract recording.wav --query "gunshots" --gain 20
  audioforge remove recording.wav --query "human speech"
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze audio contents")
    analyze_parser.add_argument("input", help="Input audio file")
    analyze_parser.add_argument(
        "--threshold", type=float, default=0.1, help="Detection confidence threshold"
    )

    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract specific sound")
    extract_parser.add_argument("input", help="Input audio file")
    extract_parser.add_argument("--query", "-q", required=True, help="Text query for extraction")
    extract_parser.add_argument("--output", "-o", help="Output file path")
    extract_parser.add_argument("--gain", type=float, default=0, help="Gain in dB")
    extract_parser.add_argument("--device", default="auto", help="Device: cuda, cpu, auto")

    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove specific sound")
    remove_parser.add_argument("input", help="Input audio file")
    remove_parser.add_argument("--query", "-q", required=True, help="Sound to remove")
    remove_parser.add_argument("--output", "-o", help="Output file path")
    remove_parser.add_argument("--device", default="auto", help="Device: cuda, cpu, auto")

    # Restore command (Full Pipeline: Separate -> Restore -> Enhance)
    restore_parser = subparsers.add_parser("restore", help="Full restoration pipeline (Separate -> VoiceFixer -> Enhance)")
    restore_parser.add_argument("input", help="Input audio file")
    restore_parser.add_argument("--query", "-q", required=True, help="What to extract")
    restore_parser.add_argument("--output", "-o", help="Output file path")
    restore_parser.add_argument("--gain", type=float, default=10, help="Final gain in dB")
    restore_parser.add_argument("--device", default="auto", help="Device: cuda, cpu, auto")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show audio file info")
    info_parser.add_argument("input", help="Input audio file")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "analyze":
            cmd_analyze(args)
        elif args.command == "extract":
            cmd_extract(args)
        elif args.command == "restore":
            cmd_restore(args)
        elif args.command == "remove":
            cmd_remove(args)
        elif args.command == "info":
            cmd_info(args)
    except KeyboardInterrupt:
        console.print("\n[red]Interrupted by user[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.exception("Command failed")
        sys.exit(1)


def cmd_analyze(args):
    """Analyze audio and show detected sounds"""
    console.print(Panel.fit("🔍 [bold blue]AudioForge Analyzer[/bold blue]"))

    analyzer = AudioAnalyzer()
    
    with console.status("[bold green]Loading model...[/bold green]"):
        analyzer.load()

    with console.status(f"[bold green]Analyzing {args.input}...[/bold green]"):
        results = analyzer.analyze(args.input, threshold=args.threshold)

    if not results:
        console.print("[yellow]No sounds detected above threshold[/yellow]")
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Sound", style="cyan")
    table.add_column("Confidence", justify="right")
    table.add_column("Bar", justify="left")

    for detection in results:
        confidence = detection["confidence"]
        bar = "█" * int(confidence * 20)
        color = "green" if confidence > 0.5 else "yellow" if confidence > 0.2 else "red"
        table.add_row(
            detection["sound"],
            f"[{color}]{confidence:.3f}[/{color}]",
            f"[{color}]{bar}[/{color}]",
        )

    console.print(table)
    console.print(f"\n[dim]Detected {len(results)} sounds above threshold {args.threshold}[/dim]")


def cmd_extract(args):
    """Extract a specific sound from audio"""
    console.print(Panel.fit("🎧 [bold blue]AudioForge Extractor[/bold blue]"))

    separator = SourceSeparator(device=args.device)

    console.print("[bold green]Loading AudioSep model...[/bold green]")
    separator.load()

    console.print(f"[bold]Input:[/bold] {args.input}")
    console.print(f"[bold]Query:[/bold] [cyan]{args.query}[/cyan]")
    console.print(f"[bold]Device:[/bold] {separator.device}")

    console.print("[bold green]Separating audio...[/bold green]")
    stem = separator.extract(args.input, args.query)

    if args.gain > 0:
        amplifier = StemAmplifier()
        console.print(f"[bold green]Amplifying by {args.gain}dB...[/bold green]")
        stem = amplifier.boost(stem, args.gain)
        console.print(f"[green]✓[/green] Amplified by {args.gain}dB")

    # Generate output filename
    if not args.output:
        input_path = Path(args.input)
        suffix = args.query.replace(" ", "_")[:20].lower()
        args.output = str(input_path.parent / f"{input_path.stem}_{suffix}_extracted.wav")

    separator.save(stem, args.output)
    console.print(f"[green]✓[/green] Saved: [bold]{args.output}[/bold]")
    console.print(f"[dim]Duration: {len(stem) / 32000:.2f} seconds[/dim]")


def cmd_restore(args):
    """Full restoration pipeline: separate → restore → enhance"""
    console.print(Panel.fit("✨ [bold blue]AudioForge Restoration Pipeline[/bold blue]"))

    # Step 1: Separate
    console.print("\n[bold]Step 1:[/bold] Separating target sound with AudioSep...")
    separator = SourceSeparator(device=args.device)
    separator.load()
    stem = separator.extract(args.input, args.query)

    # Step 2: Restore with VoiceFixer
    console.print("[bold]Step 2:[/bold] AI restoration & reconstruction with VoiceFixer...")
    from core.restorer import AudioRestorer
    restorer = AudioRestorer(device=args.device)
    restorer.load()

    # VoiceFixer operates at 44100 Hz
    import librosa
    import soundfile as sf
    stem_44k = librosa.resample(stem, orig_sr=32000, target_sr=44100)
    restored = restorer.restore(stem_44k, sr=44100)

    # Step 3: Amplify
    if args.gain > 0:
        amplifier = StemAmplifier()
        restored = amplifier.boost(restored, args.gain)
        console.print(f"[green]✓[/green] Amplified by {args.gain}dB")

    # Step 4: Final enhancement
    amplifier = StemAmplifier()
    restored = amplifier.prepare_for_analysis(restored, 44100)

    # Save
    if not args.output:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_restored.wav")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, restored, 44100)
    console.print(f"[green]✓[/green] Saved: [bold]{args.output}[/bold]")
    console.print(f"[dim]Duration: {len(restored) / 44100:.2f} seconds (44.1kHz High-Fidelity)[/dim]")


def cmd_remove(args):
    """Remove a specific sound from audio"""
    console.print(Panel.fit("🔇 [bold blue]AudioForge Remover[/bold blue]"))

    separator = SourceSeparator(device=args.device)

    with console.status("[bold green]Loading models...[/bold green]"):
        separator.load()

    console.print(f"[bold]Input:[/bold] {args.input}")
    console.print(f"[bold]Removing:[/bold] [cyan]{args.query}[/cyan]")

    # Extract the sound to remove, then subtract from original
    with console.status("[bold green]Separating audio...[/bold green]"):
        stem_to_remove = separator.extract(args.input, args.query)
        original = separator.load_audio(args.input)
        remaining = original - stem_to_remove

    if not args.output:
        input_path = Path(args.input)
        suffix = "no_" + args.query.replace(" ", "_")[:15]
        args.output = str(input_path.parent / f"{input_path.stem}_{suffix}.wav")

    separator.save(remaining, args.output)
    console.print(f"[green]✓[/green] Saved: [bold]{args.output}[/bold]")


def cmd_info(args):
    """Show audio file information"""
    import librosa

    console.print(Panel.fit("ℹ️  [bold blue]Audio Info[/bold blue]"))

    try:
        y, sr = librosa.load(args.input, sr=None, mono=False)
        duration = librosa.get_duration(y=y, sr=sr)

        info_table = Table(show_header=False)
        info_table.add_column("Property", style="bold cyan")
        info_table.add_column("Value")

        info_table.add_row("File", args.input)
        info_table.add_row("Duration", f"{duration:.2f} seconds")
        info_table.add_row("Sample Rate", f"{sr} Hz")
        info_table.add_row("Channels", "Mono" if y.ndim == 1 else f"Stereo ({y.shape[0]}ch)")
        info_table.add_row("Samples", f"{y.shape[-1]:,}")

        import os
        size_mb = os.path.getsize(args.input) / 1024 / 1024
        info_table.add_row("File Size", f"{size_mb:.2f} MB")

        console.print(info_table)

    except Exception as e:
        console.print(f"[red]Failed to load audio: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
