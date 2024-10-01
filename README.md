# RTSP Stream Processing with OpenCV

This project illustrates the process of capturing an RTSP stream, processing it with OpenCV, and displaying the resultant video to the user. It serves as a example for developers.

## Features

- Captures video from an RTSP source (H.264).
- Processes frames using OpenCV (converts to grayscale).
- Displays the processed video stream.
- Supports both real RTSP sources and a test video source.

## Requirements

- Python 3.12
- Poetry for dependency management

## Installation

1. Clone this repository
2. Install dependencies using Poetry:

```bash
poetry env use python3.12
poetry install
```

## Usage

Run the script using Poetry:

```bash
poetry run python pipeline.py
```


### Command-line options:

- `--fake-source`: Use a test video source instead of RTSP
- `--src-uri`: Specify the RTSP source URI (default: rtsp://127.0.0.1:8554/test)
- `--create-graph`: Generate a visual representation of the GStreamer pipeline
- `--debug`: Enable debug logging

## Example

To run with a custom RTSP source:

```bash
poetry run python pipeline.py --src-uri rtsp://127.0.0.1:8554/test
```

To use a fake source (test video):

```bash
poetry run python pipeline.py --fake-source
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

This is merely a sample and not a real product; therefore, I do not anticipate any external contributions.
