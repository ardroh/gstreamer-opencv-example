# MIT License
#
# Copyright (c) 2024 ardroh
#
# Permission is granted to use, copy, modify, and distribute this software with
# attribution to the original author.

import argparse
from enum import Enum
import logging
import os
from typing import Optional
import cv2
import gi
import numpy as np

gi.require_version('Gst', '1.0')  # noqa: E402
from gi.repository import Gst, GLib  # noqa: E402

LOGGING_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
DEFAULT_RTSP_URI = "rtsp://127.0.0.1:8554/test"
DEFAULT_FRAMERATE = "30/1"


def setup_logging(debug=False):
    logging_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=logging_level,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

global logger
logger = None


# Initialize GStreamer
Gst.init(None)


def main():
    parser = argparse.ArgumentParser(description="Video processing pipeline")
    parser.add_argument('--input-mode', '--im', type=str,
                        choices=['rtsp', 'file', 'fake'], default='fake',
                        help='Input mode for the pipeline')
    parser.add_argument('--src-uri', '-s', type=str,
                        default=DEFAULT_RTSP_URI, help='Source URI for RTSP. Only used if input-mode is "rtsp".')
    parser.add_argument('--input-file', '-f', type=str,
                        help='File path for video file. Only used if input-mode is "file".')
    parser.add_argument('--create-graph', action='store_true',
                        help='Create graph of the pipeline')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    args = parser.parse_args()

    global logger
    logger = setup_logging(args.debug)

    pipeline = Gst.Pipeline.new('video-processing-pipeline')

    # === CONVERTER FOR OPENCV ===
    # Converts the video stream to a format that can be used by the sink (OpenCV in this case).
    # https://gstreamer.freedesktop.org/documentation/videoconvertscale/videoconvert.html?gi-language=c#videoconvert-page
    converter_opencv = Gst.ElementFactory.make(
        'videoconvert', 'converter-opencv')
    pipeline.add(converter_opencv)

    source_class: Optional[BaseSource] = None
    if args.input_mode == 'fake':
        source_class = FakeSource()
    elif args.input_mode == 'file':
        if not args.input_file:
            raise ValueError("File path is required for file input mode")
        source_class = FileSource(args.input_file)
    elif args.input_mode == 'rtsp':
        if not args.src_uri:
            raise ValueError("Source URI is required for RTSP input mode")
        source_class = RTSPSource(args.src_uri)
    else:
        raise ValueError(f"Invalid input mode: {args.input_mode}")

    configure_source_bin(source_class, pipeline, converter_opencv)

    # === APPSINK ===
    # This sink will receive the decoded frames and push them for OpenCV processing.
    # After the frames are processed by OpenCV, they will be pushed back into the pipeline
    # via emitting the "push-sample" signal on the appsrc element.
    # https://gstreamer.freedesktop.org/documentation/app/appsink.html?gi-language=c
    appsink = Gst.ElementFactory.make("appsink", "opencv-sink")
    # Caps based from the example: https://gist.github.com/cbenhagen/76b24573fa63e7492fb6
    caps = Gst.caps_from_string(
        "video/x-raw, format=(string){BGR, GRAY8}; video/x-bayer,format=(string){rggb,bggr,grbg,gbrg}")
    appsink.set_property("caps", caps)
    appsink.set_property("emit-signals", True)
    appsink.set_property("max-buffers", 1)
    appsink.set_property("drop", True)

    # === APPSRC ===
    # This source will receive the processed frames from OpenCV and push them back into the pipeline.
    # https://gstreamer.freedesktop.org/documentation/app/appsrc.html?gi-language=c
    appsrc = Gst.ElementFactory.make("appsrc", "opencv-src")
    appsrc.set_property("is-live", True)
    # Connect the new-sample signal
    appsink.connect("new-sample", on_new_sample, appsrc)

    # === CONVERTER FOR SINK ===
    # Converts the video stream to a format that can be used by the sink (autovideosink in this case).
    converter_display = Gst.ElementFactory.make(
        'videoconvert', 'converter-sink')

    # == SINK ==
    # Displays the video stream.
    # https://gstreamer.freedesktop.org/documentation/autodetect/autovideosink.html?gi-language=c#autovideosink-page
    display_sink = Gst.ElementFactory.make('autovideosink', 'sink')
    display_sink.set_property("async-handling", True)

    pipeline.add(appsink)
    pipeline.add(appsrc)
    pipeline.add(converter_display)
    pipeline.add(display_sink)

    # converter_opencv is already linked to the source bin inside configure_source_bin()
    Gst.Element.link(converter_opencv, appsink)
    Gst.Element.link(appsrc, converter_display)
    Gst.Element.link(converter_display, display_sink)

    if args.create_graph:
        create_pipeline_graph(pipeline, "pipeline_initial")

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_bus_message, pipeline)
    bus.connect("message::error", on_error, pipeline)

    # Start the pipeline
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        logger.error("Failed to set the pipeline to the playing state")
        return
    elif ret == Gst.StateChangeReturn.ASYNC:
        logger.info("Pipeline is ASYNC, waiting for PLAYING state...")
        ret, state, pending = pipeline.get_state(Gst.CLOCK_TIME_NONE)
        if ret != Gst.StateChangeReturn.SUCCESS:
            logger.error(f"Failed to reach PLAYING state. Current state: {
                         state}, Pending: {pending}")
            return

    logger.info("Pipeline is now PLAYING")

    if args.create_graph:
        create_pipeline_graph(pipeline, "pipeline_playing")

    try:
        loop = GLib.MainLoop()
        loop.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected, stopping the pipeline")
    finally:
        pipeline.set_state(Gst.State.NULL)


class BaseSource:
    def __init__(self):
        pass


class FakeSource(BaseSource):
    """
    Fake source class (videotestsrc).
    """

    def __init__(self):
        pass


class RTSPSource(BaseSource):
    """
    RTSP source class.

    Args:
        src_uri (str): The URI of the RTSP server.
    """

    def __init__(self, src_uri: str):
        self.src_uri = src_uri


class FileSource(BaseSource):
    """
    File source class.

    Args:
        file_path (str): The path to the video file.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path


def configure_source_bin(source_class: BaseSource, pipeline: Gst.Pipeline, next_pad: Gst.Element) -> Gst.Element:
    """
    Configure the source bin for the pipeline. All source elements will added to a pipeline and linked together.
    The resulting source bin will be linked to the next pad of the pipeline.

    Args:
        source_class (BaseSource): The source class to configure.
        pipeline (Gst.Pipeline): The pipeline to configure.
    """
    if isinstance(source_class, FakeSource):
        source = Gst.ElementFactory.make('videotestsrc', 'source')
        pipeline.add(source)
        Gst.Element.link(source, next_pad)
    elif isinstance(source_class, RTSPSource):
        # === SOURCE ===
        # Reads the video stream from the RTSP server.
        # https://gstreamer.freedesktop.org/documentation/rtsp/rtspsrc.html?gi-language=c
        source = Gst.ElementFactory.make('rtspsrc', 'rtsp-source')
        source.set_property('location', source_class.src_uri)
        # === H264 EXTRACTOR ===
        # Extracts the H264 packets from the RTP stream.
        # https://gstreamer.freedesktop.org/documentation/rtp/rtph264depay.html?gi-language=c
        h264_extractor = Gst.ElementFactory.make(
            'rtph264depay', 'h264-extractor')
        # === PARSER ===
        # Prepares the H264 packets into a format that can be used for decoding.
        # https://gstreamer.freedesktop.org/documentation/videoparsersbad/h264parse.html?gi-language=c
        parser = Gst.ElementFactory.make('h264parse', 'parser')
        # === DECODER ===
        # Decodes the H264 packets into a raw video frames for processing or displaying.
        # https://gstreamer.freedesktop.org/documentation/videoparsersbad/h264parse.html?gi-language=c
        decoder = Gst.ElementFactory.make('avdec_h264', 'decoder')
        source.connect("pad-added", on_pad_added_rtsp, h264_extractor)
        pipeline.add(source)
        pipeline.add(h264_extractor)
        pipeline.add(parser)
        pipeline.add(decoder)
        Gst.Element.link(source, h264_extractor)
        Gst.Element.link(h264_extractor, parser)
        Gst.Element.link(parser, decoder)
        Gst.Element.link(decoder, next_pad)
    elif isinstance(source_class, FileSource):
        logger.info(f"Using file source: {source_class.file_path}")
        if not os.path.exists(source_class.file_path):
            raise FileNotFoundError(f"File not found: {source_class.file_path}")
        source = Gst.ElementFactory.make('filesrc', 'file-source')
        source.set_property('location', source_class.file_path)
        decoder = Gst.ElementFactory.make('decodebin', 'decoder')
        # Once capabilities are negotiated, the pad-added signal will be emitted.
        decoder.connect("pad-added", on_pad_added_decodebin, next_pad)
        pipeline.add(source)
        pipeline.add(decoder)
        Gst.Element.link(source, decoder)
    else:
        raise ValueError(f"Invalid source class: {source_class}")


def on_new_sample(sink: Gst.Element, appsrc: Gst.Element) -> Gst.FlowReturn:
    """
    Callback function for handling new samples received by the appsink.

    This function will process the sample and push it to the appsrc.
    """
    logger.debug("New sample received from sink")
    sample = sink.emit("pull-sample")
    if sample:
        new_sample = process_frame(sample)
        if new_sample:
            ret = appsrc.emit("push-sample", new_sample)
            if ret != Gst.FlowReturn.OK:
                logger.error(f"Failed to push sample: {ret}")
            return Gst.FlowReturn.OK
    else:
        logger.error("Failed to pull sample")
    return Gst.FlowReturn.ERROR


def process_frame(sample: Gst.Sample) -> Gst.Sample:
    """
    Process a video frame from the GStreamer pipeline.

    Converts the input frame to grayscale and then back to RGB format.
    This results in a black and white image while maintaining the RGB structure.

    Args:
        sample (Gst.Sample): The input video frame as a GStreamer Sample.

    Returns:
        Gst.Sample: The processed black and white frame as a new GStreamer Sample.
    """
    buf = sample.get_buffer()
    caps = sample.get_caps()
    # Get width and height of the image
    width = caps.get_structure(0).get_value('width')
    height = caps.get_structure(0).get_value('height')
    image_format = caps.get_structure(0).get_value('format')
    logger.debug(f"Processing new sample with dimensions - Width: {
                 width}, Height: {height}, Format: {image_format}")
    # Convert Gst.Buffer to np.ndarray
    buffer = buf.extract_dup(0, buf.get_size())
    if image_format == 'I420':
        img_array = np.frombuffer(buffer, dtype=np.uint8).reshape(
            (int(height*1.5), width))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_YUV2RGB_I420)
    else:
        img_array = np.ndarray(
            (height, width, 3),
            buffer=buffer,
            dtype=np.uint8
        )
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    # Convert back to RGB (but it will be black and white)
    bw_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    # Create new caps for the processed frame
    new_caps = Gst.Caps.from_string(
        f"video/x-raw,format=RGB,width={width},height={height},framerate={DEFAULT_FRAMERATE}"
    )
    # Create a new Gst.Buffer
    new_buf = Gst.Buffer.new_wrapped(bw_img.tobytes())
    # Create a new sample with the processed buffer and new caps
    new_sample = Gst.Sample.new(new_buf, new_caps, None, None)
    return new_sample


def on_pad_added_rtsp(src: Gst.Element, new_pad: Gst.Pad, h264_extractor: Gst.Element) -> None:
    """
    Callback function for handling dynamically added pads.

    Args:
        src (Gst.Element): The source element that created the new pad.
        new_pad (Gst.Pad): The newly created pad.
        h264_extractor (Gst.Element): The h264 extractor element to link to.
    """
    caps = new_pad.get_current_caps()
    name = caps.to_string()
    logger.info(
        f"Received new pad '{new_pad.name}' with caps '{name}' from '{src.name}'")
    logger.info(f"Pad caps: {new_pad.get_current_caps().to_string()}")
    if "application/x-rtp" in name and "encoding-name=(string)H264" in name:
        sink_pad = h264_extractor.get_static_pad('sink')
        if not sink_pad.is_linked():
            ret = new_pad.link(sink_pad)
            if ret == Gst.PadLinkReturn.OK:
                logger.info("Pad linked successfully")
            else:
                logger.error(f"Pad link failed with error {ret}")
        else:
            logger.info("Sink pad is already linked")
    else:
        logger.info(
            "Pad does not have 'application/x-rtp' with H264 encoding, ignoring.")


def on_pad_added_decodebin(decoder: Gst.Element, new_pad: Gst.Pad, next_element: Gst.Element) -> None:
    """
    Callback function for handling dynamically added pads from decodebin.

    Args:
        decoder (Gst.Element): The decodebin element.
        new_pad (Gst.Pad): The newly created pad.
        next_element (Gst.Element): The next element to link to.
    """
    sink_pad = next_element.get_static_pad('sink')
    if not sink_pad.is_linked():
        ret = new_pad.link(sink_pad)
        if ret == Gst.PadLinkReturn.OK:
            logger.info("Pad linked successfully")
        else:
            logger.error(f"Pad link failed with error {ret}")


def on_bus_message(bus, message, pipeline):
    """
    Callback function for handling messages from the GStreamer pipeline bus.
    """
    t = message.type
    if t == Gst.MessageType.EOS:
        logger.info("End of stream")
        pipeline.set_state(Gst.State.NULL)
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        logger.error("Error in the pipeline: %s: %s", err, debug)
        pipeline.set_state(Gst.State.NULL)
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        logger.warning("Warning in the pipeline: %s: %s", err, debug)
    elif t == Gst.MessageType.INFO:
        info, debug = message.parse_info()
        logger.info("Info in the pipeline: %s: %s", info, debug)
    elif t == Gst.MessageType.STATE_CHANGED:
        old_state, new_state, pending_state = message.parse_state_changed()
        logger.info("State changed from %s to %s", old_state, new_state)
    else:
        logger.debug("Received message of type %s", t)


def on_error(bus, message, pipeline):
    """
    Callback function for handling error messages from the GStreamer pipeline bus.
    """
    err, debug = message.parse_error()
    logger.error("Error in the pipeline: %s: %s", err, debug)
    pipeline.set_state(Gst.State.NULL)


def create_pipeline_graph(pipeline, filename):
    """
    Create a graph of the pipeline and save it as a PNG image.
    """
    Gst.debug_bin_to_dot_file(pipeline, Gst.DebugGraphDetails.ALL, filename)
    os.system(f"dot -Tpng {filename}.dot -o {filename}.png")
    logger.info(f"Pipeline graph saved to {filename}.png")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"{e}")
        import traceback
        traceback_str = traceback.format_exc()
        logger.debug(traceback_str)
