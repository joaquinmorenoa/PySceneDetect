# -*- coding: utf-8 -*-
#
#         PySceneDetect: Python-Based Video Scene Detector
#   ---------------------------------------------------------------
#     [  Site: http://www.bcastell.com/projects/PySceneDetect/   ]
#     [  Github: https://github.com/Breakthrough/PySceneDetect/  ]
#     [  Documentation: http://pyscenedetect.readthedocs.org/    ]
#
# Copyright (C) 2014-2020 Brandon Castellano <http://www.bcastell.com>.
#
# PySceneDetect is licensed under the BSD 3-Clause License; see the included
# LICENSE file, or visit one of the following pages for details:
#  - https://github.com/Breakthrough/PySceneDetect/
#  - http://www.bcastell.com/projects/PySceneDetect/
#
# This software uses Numpy, OpenCV, click, tqdm, simpletable, and pytest.
# See the included LICENSE files or one of the above URLs for more information.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

""" Module: ``scenedetect.detectors.content_detector``

This module implements the :py:class:`ContentDetector`, which compares the
difference in content between adjacent frames against a set threshold/score,
which if exceeded, triggers a scene cut.

This detector is available from the command-line interface by using the
`detect-content` command.
"""

# Third-Party Library Imports
import numpy
import cv2

# PySceneDetect Library Imports
from scenedetect.scene_detector import SceneDetector


def rgb_to_hsv(frame):
    return cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))

def calculate_frame_delta(curr_frame, last_frame):
    delta_hsv = [0, 0, 0, 0]
    for i in range(3):
        num_pixels = curr_frame[i].shape[0] * curr_frame[i].shape[1]
        curr_frame[i] = curr_frame[i].astype(numpy.int32)
        last_frame[i] = last_frame[i].astype(numpy.int32)
        delta_hsv[i] = numpy.sum(
            numpy.abs(curr_frame[i] - last_frame[i])) / float(num_pixels)
    delta_hsv[3] = sum(delta_hsv[0:3]) / 3.0
    return delta_hsv


class ContentDetector(SceneDetector):
    """Detects fast cuts using changes in colour and intensity between frames.

    Since the difference between frames is used, unlike the ThresholdDetector,
    only fast cuts are detected with this method.  To detect slow fades between
    content scenes still using HSV information, use the DissolveDetector.
    """

    def __init__(self, threshold=30.0, min_scene_len=15):
        # type: (float, Union[int, FrameTimecode]) -> None
        super(ContentDetector, self).__init__()
        self.threshold = threshold
        # Minimum length of any given scene, in frames (int) or FrameTimecode
        self.min_scene_len = min_scene_len
        self.last_frame = None
        self.last_scene_cut = None
        self.last_hsv = None
        self._metric_keys = ['content_val', 'delta_hue', 'delta_sat', 'delta_lum']
        self.cli_name = 'detect-content'
        self._new_scene_cut = None
        self.flicker_frames = 0

        #assert self.min_scene_len > self.flicker_frames


    def process_frame(self, frame_num, frame_img):
        # type: (int, numpy.ndarray) -> List[int]
        """ Similar to ThresholdDetector, but using the HSV colour space DIFFERENCE instead
        of single-frame RGB/grayscale intensity (thus cannot detect slow fades with this method).

        Arguments:
            frame_num (int): Frame number of frame that is being passed.

            frame_img (Optional[int]): Decoded frame image (numpy.ndarray) to perform scene
                detection on. Can be None *only* if the self.is_processing_required() method
                (inhereted from the base SceneDetector class) returns True.

        Returns:
            List[int]: List of frames where scene cuts have been detected. There may be 0
            or more frames in the list, and not necessarily the same as frame_num.
        """

        cut_list = []
        metric_keys = self._metric_keys
        _unused = ''

        # Initialize last scene cut point at the beginning of the frames of interest.
        if self.last_scene_cut is None:
            self.last_scene_cut = frame_num

        # Handle flash suppression (i.e. validate a previously detected scene transition).
        #
        # TODO: Test rapid toggling.
        #
        # TODO: Add to statsfile (requires new column?, store N/A if no value,
        #                         or delta from place where data starts otherwise)
        #
        if self._new_scene_cut is not None:
            # Make sure the next N [flicker_frames] frames are also above the threshold.
            if (frame_num - self._new_scene_cut[0]) > self.flicker_frames:
                # We passed the threshold!  (if ff = 0 should have same behavior as current)
                #                           (if ff = 1 should ignore a 1-frame flash)
                #                           (if ff = 2 should ignore a 2-frame flash, etc)
                cut_list.append(self._new_scene_cut[0])
                self.last_scene_cut = self._new_scene_cut[0]
                self._new_scene_cut = None

            # This has to be cleaned up a lot.

            # Check if delta between last_frame and frame_img still exceeds threshold.
            # if yes, do nothing so the branch above eventually fires.
            #if not, clear the cut, because it was a flash.
            curr_hsv = rgb_to_hsv(frame_img)
            last_hsv = rgb_to_hsv(self._new_scene_cut[0])
            delta_h, delta_s, delta_v, delta_hsv_avg = calculate_frame_delta(
                curr_hsv, last_hsv)
            if delta_hsv_avg < self.threshold:
                # Returned to the same scene within threshold; ignore.
                self._new_scene_cut = None



        # We can only start detecting once we have a frame to compare with.
        elif self.last_frame is not None:
            # Change in average of HSV (hsv), (h)ue only, (s)aturation only, (l)uminance only.
            # These are refered to in a statsfile as their respective self._metric_keys string.
            delta_hsv_avg, delta_h, delta_s, delta_v = 0.0, 0.0, 0.0, 0.0

            if (self.stats_manager is not None and
                    self.stats_manager.metrics_exist(frame_num, metric_keys)):
                delta_hsv_avg, delta_h, delta_s, delta_v = self.stats_manager.get_metrics(
                    frame_num, metric_keys)

            else:
                curr_hsv = rgb_to_hsv(frame_img)
                last_hsv = self.last_hsv
                if not last_hsv:
                    last_hsv = rgb_to_hsv(self.last_frame)
                delta_h, delta_s, delta_v, delta_hsv_avg = calculate_frame_delta(
                    curr_hsv, last_hsv)

                if self.stats_manager is not None:
                    self.stats_manager.set_metrics(frame_num, {
                        metric_keys[0]: delta_hsv_avg,
                        metric_keys[1]: delta_h,
                        metric_keys[2]: delta_s,
                        metric_keys[3]: delta_v})

                self.last_hsv = curr_hsv

            # We consider any frame over the threshold a new scene, but only if
            # the minimum scene length has been reached (otherwise it is ignored).
            if delta_hsv_avg >= self.threshold and (
                    (frame_num - self.last_scene_cut) >= self.min_scene_len):

                # Move into 'validating scene transition' mode - wait until
                # the next frame - make sure that one also has a delta
                # above the threshold.
                #
                # Might need a state machine, heh.
                #
                if self.flicker_frames > 0:
                    self._new_scene_cut = (frame_num, self.last_frame)
                else:
                    cut_list.append(frame_num)
                    self.last_scene_cut = frame_num

            if self.last_frame is not None and self.last_frame is not _unused:
                del self.last_frame
                self.last_frame = None

        # If we have the next frame computed, don't copy the current frame
        # into last_frame since we won't use it on the next call anyways.
        if (self.stats_manager is not None and
                self.stats_manager.metrics_exist(frame_num+1, metric_keys)):
            self.last_frame = _unused
        else:
            self.last_frame = frame_img.copy()

        return cut_list


    #def post_process(self, frame_num):
    #    """ TODO: Based on the parameters passed to the ContentDetector constructor,
    #        ensure that the last scene meets the minimum length requirement,
    #        otherwise it should be merged with the previous scene.
    #    """
    #    return []
