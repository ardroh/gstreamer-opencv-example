# MIT License
#
# Copyright (c) 2024 ardroh
#
# Permission is granted to use, copy, modify, and distribute this software with
# attribution to the original author.

import argparse
import logging
import os

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
    parser.add_argument('--fake-source', action='store_true',
                        help='Use fake source instead of RTSP')
    parser.add_argument('--src-uri', '-s', type=str,
                        default=DEFAULT_RTSP_URI, help='Source URI for RTSP')
    parser.add_argument('--create-graph', action='store_true',
                        help='Create graph of the pipeline')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    args = parser.parse_args()

    global logger
    logger = setup_logging(args.debug)

    pipeline = Gst.Pipeline.new('video-processing-pipeline')

    if args.fake_source:
        source = Gst.ElementFactory.make('videotestsrc', 'source')
    else:
        # === SOURCE ===
        # Reads the video stream from the RTSP server.
        # https://gstreamer.freedesktop.org/documentation/rtsp/rtspsrc.html?gi-language=c
        source = Gst.ElementFactory.make('rtspsrc', 'rtsp-source')
        source.set_property('location', args.src_uri)
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

    # === CONVERTER FOR OPENCV ===
    # Converts the video stream to a format that can be used by the sink (OpenCV in this case).
    # https://gstreamer.freedesktop.org/documentation/videoconvertscale/videoconvert.html?gi-language=c#videoconvert-page
    converter_opencv = Gst.ElementFactory.make(
        'videoconvert', 'converter-opencv')

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

    if not args.fake_source:
        pipeline.add(source)
        pipeline.add(h264_extractor)
        pipeline.add(parser)
        pipeline.add(decoder)
    pipeline.add(converter_display)
    pipeline.add(converter_opencv)
    pipeline.add(appsink)
    pipeline.add(appsrc)
    pipeline.add(display_sink)

    # Link the elements
    if args.fake_source:
        Gst.Element.link(source, converter_opencv)
    else:
        # We can't link source directly because it creates pads dynamically
        source.connect("pad-added", on_pad_added, h264_extractor)
        Gst.Element.link(source, h264_extractor)
        Gst.Element.link(h264_extractor, parser)
        Gst.Element.link(parser, decoder)
        Gst.Element.link(decoder, converter_opencv)
    Gst.Element.link(converter_opencv, appsink)
    Gst.Element.link(appsrc, converter_display)
    Gst.Element.link(converter_display, display_sink)

    if args.create_graph:
        create_pipeline_graph(pipeline, "pipeline_initial")

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

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_bus_message, pipeline)

    try:
        loop = GLib.MainLoop()
        loop.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected, stopping the pipeline")
    finally:
        pipeline.set_state(Gst.State.NULL)


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


def on_pad_added(src: Gst.Element, new_pad: Gst.Pad, h264_extractor: Gst.Element) -> None:
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


def create_pipeline_graph(pipeline, filename):
    """
    Create a graph of the pipeline and save it as a PNG image.
    """
    Gst.debug_bin_to_dot_file(pipeline, Gst.DebugGraphDetails.ALL, filename)
    os.system(f"dot -Tpng {filename}.dot -o {filename}.png")
    logger.info(f"Pipeline graph saved to {filename}.png")


if __name__ == '__main__':
    main()