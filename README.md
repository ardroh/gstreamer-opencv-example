# RTSP Stream Processing with OpenCV

This project illustrates the process of capturing an RTSP stream, processing it with OpenCV, and displaying the resultant video to the user. It serves as a example for developers.

## Features

- Supported input sources:
  - RTSP (H.264)
  - Video file
  - Fake source (test video)
- Processes frames using OpenCV (converts to grayscale, adds timestamp).
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

```
$ poetry run python pipeline.py --help
usage: pipeline.py [-h] [--input-mode {rtsp,file,fake}] [--src-uri SRC_URI] [--input-file INPUT_FILE]
                   [--black-and-white] [--timestamp] [--create-graph] [--debug]

Video processing pipeline

options:
  -h, --help            show this help message and exit
  --input-mode {rtsp,file,fake}, --im {rtsp,file,fake}
                        Input mode for the pipeline
  --src-uri SRC_URI, -s SRC_URI
                        Source URI for RTSP. Only used if input-mode is "rtsp".
  --input-file INPUT_FILE, -f INPUT_FILE
                        File path for video file. Only used if input-mode is "file".
  --black-and-white, --bw
                        Convert the video to black and white
  --timestamp, --ts     Add timestamp to the video
  --create-graph        Create graph of the pipeline
  --debug               Enable debug logging
```

## Example

To run with a custom RTSP source:

```bash
poetry run python pipeline.py --input-mode rtsp --src-uri rtsp://127.0.0.1:8554/test
```

To use a fake source (test video):

```bash
poetry run python pipeline.py --input-mode fake
```

To run with a video file:

```bash
poetry run python pipeline.py --input-mode file --input-file ./resources/test.mp4
```

## Visualization

To visualize the pipeline, run the script with the `--create-graph` option.

### Initial State (RTSP)


![Initial State](./docs/rtsp_pipeline_initial.png)

### Playing (RTSP)

![Playing](./docs/rtsp_pipeline_playing.png)

## Change Log

See [CHANGELOG.md](./CHANGELOG.md) for details.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

This is merely a sample and not a real product; therefore, I do not anticipate any external contributions.
