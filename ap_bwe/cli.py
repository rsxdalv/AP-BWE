"""Command-line interface for AP-BWE package."""

import argparse
import sys

def main():
    """Main CLI entrypoint for AP-BWE."""
    parser = argparse.ArgumentParser(
        description='AP-BWE: Speech Bandwidth Extension with Parallel Amplitude and Phase Prediction',
        usage='''ap-bwe <command> [<args>]

Available commands:
   train-16k         Train a 16kHz bandwidth extension model
   train-48k         Train a 48kHz bandwidth extension model
   inference-16k     Generate high-quality 16kHz audio from low-resolution input
   inference-48k     Generate high-quality 48kHz audio from low-resolution input
''')
    parser.add_argument('command', help='Command to run')
    
    # Parse args
    args = parser.parse_args(sys.argv[1:2])
    
    # Route to appropriate module
    if args.command == 'train-16k':
        from ap_bwe.train.train_16k import main as train_main
        # Pass the remaining arguments, excluding the command itself
        train_main(sys.argv[2:])
    elif args.command == 'train-48k':
        from ap_bwe.train.train_48k import main as train_main
        # Pass the remaining arguments, excluding the command itself
        train_main(sys.argv[2:])
    elif args.command == 'inference-16k':
        from ap_bwe.inference.inference_16k import main as inference_main
        # Pass the remaining arguments, excluding the command itself
        inference_main(sys.argv[2:])
    elif args.command == 'inference-48k':
        from ap_bwe.inference.inference_48k import main as inference_main
        # Pass the remaining arguments, excluding the command itself
        inference_main(sys.argv[2:])
    else:
        print(f'Unrecognized command: {args.command}')
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
