#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.2a1),
    on Sat Oct 12 15:25:31 2024
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '4'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware, iohub
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from eeg
import pyxid2
import threading
import signal


def exit_after(s):
    '''
    function decorator to raise KeyboardInterrupt exception
    if function takes longer than s seconds
    '''
    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(s, signal.raise_signal, args=[signal.SIGINT])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result
        return inner
    return outer


@exit_after(1)  # exit if function takes longer than 1 seconds
def _get_xid_devices():
    return pyxid2.get_xid_devices()


def get_xid_devices():
    print("Getting a list of all attached XID devices...")
    attempt_count = 0
    while attempt_count >= 0:
        attempt_count += 1
        print('     Attempt:', attempt_count)
        attempt_count *= -1  # try to exit the while loop
        try:
            devices = _get_xid_devices()
        except KeyboardInterrupt:
            attempt_count *= -1  # get back in the while loop
    return devices


devices = get_xid_devices()

if devices:
    dev = devices[0]
    print("Found device:", dev)
    assert dev.device_name == 'Cedrus C-POD', "Incorrect C-POD detected."
    dev.set_pulse_duration(50)  # set pulse duration to 50ms

    # Start EEG recording
    print("Sending trigger code 126 to start EEG recording...")
    dev.activate_line(bitmask=126)  # trigger 126 will start EEG
    print("Waiting 10 seconds for the EEG recording to start...")
    print("")
    core.wait(10)  # wait 10s for the EEG system to start recording

    # Marching lights test
    print("C-POD<->eego 7-bit trigger lines test...")
    for line in range(1, 8):  # raise lines 1-7 one at a time
        print("  raising line {} (bitmask {})".format(line, 2 ** (line-1)))
        dev.activate_line(lines=line)
        core.wait(0.5)  # wait 500ms between two consecutive triggers
    dev.con.set_digio_lines_to_mask(0)  # XidDevice.clear_all_lines()
    print("EEG system is now ready for the experiment to start.")

else:
    # Dummy XidDevice for code components to run without C-POD connected
    class dummyXidDevice(object):
        def __init__(self):
            pass
        def activate_line(self, lines=None, bitmask=None):
            pass


    print("WARNING: No C-POD connected for this session! "
          "You must start/stop EEG recording manually!")
    dev = dummyXidDevice()

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.2a1'
expName = 'object_location'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1920, 1080]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s/%s_%s_%s' % (expInfo['participant'], expInfo['participant'], expName, expInfo['session'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/alexhe/Dropbox (Personal)/Active_projects/PsychoPy/exp_object_location/object_location.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('debug')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup eyetracking
    ioConfig['eyetracker.eyelink.EyeTracker'] = {
        'name': 'tracker',
        'model_name': 'EYELINK 1000 DESKTOP',
        'simulation_mode': False,
        'network_settings': '100.1.1.1',
        'default_native_data_file_name': 'EXPFILE',
        'runtime_settings': {
            'sampling_rate': 1000.0,
            'track_eyes': 'LEFT_EYE',
            'sample_filtering': {
                'FILTER_FILE': 'FILTER_LEVEL_OFF',
                'FILTER_ONLINE': 'FILTER_LEVEL_OFF',
            },
            'vog_settings': {
                'pupil_measure_types': 'PUPIL_DIAMETER',
                'tracking_mode': 'PUPIL_CR_TRACKING',
                'pupil_center_algorithm': 'ELLIPSE_FIT',
            }
        }
    }
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    deviceManager.devices['eyetracker'] = ioServer.getDevice('tracker')
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('key_welcome') is None:
        # initialise key_welcome
        key_welcome = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_welcome',
        )
    # create speaker 'read_welcome'
    deviceManager.addDevice(
        deviceName='read_welcome',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('key_et') is None:
        # initialise key_et
        key_et = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_et',
        )
    # create speaker 'read_et'
    deviceManager.addDevice(
        deviceName='read_et',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # create speaker 'read_start'
    deviceManager.addDevice(
        deviceName='read_start',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('key_instruct_intro_1') is None:
        # initialise key_instruct_intro_1
        key_instruct_intro_1 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_instruct_intro_1',
        )
    # create speaker 'read_instruct_intro_1'
    deviceManager.addDevice(
        deviceName='read_instruct_intro_1',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('key_instruct_intro_2') is None:
        # initialise key_instruct_intro_2
        key_instruct_intro_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_instruct_intro_2',
        )
    # create speaker 'read_instruct_intro_2'
    deviceManager.addDevice(
        deviceName='read_instruct_intro_2',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('key_instruct_condition') is None:
        # initialise key_instruct_condition
        key_instruct_condition = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_instruct_condition',
        )
    # create speaker 'read_instruct_condition'
    deviceManager.addDevice(
        deviceName='read_instruct_condition',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('key_diagram_condition') is None:
        # initialise key_diagram_condition
        key_diagram_condition = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_diagram_condition',
        )
    # create speaker 'read_instruct_response'
    deviceManager.addDevice(
        deviceName='read_instruct_response',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('key_instruct_review') is None:
        # initialise key_instruct_review
        key_instruct_review = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_instruct_review',
        )
    # create speaker 'read_instruct_review'
    deviceManager.addDevice(
        deviceName='read_instruct_review',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('key_practice_repeat') is None:
        # initialise key_practice_repeat
        key_practice_repeat = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_practice_repeat',
        )
    # create speaker 'read_practice_repeat'
    deviceManager.addDevice(
        deviceName='read_practice_repeat',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('key_response_test') is None:
        # initialise key_response_test
        key_response_test = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_response_test',
        )
    if deviceManager.getDevice('key_checkpoint') is None:
        # initialise key_checkpoint
        key_checkpoint = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_checkpoint',
        )
    # create speaker 'read_checkpoint'
    deviceManager.addDevice(
        deviceName='read_checkpoint',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('key_instruct_begin') is None:
        # initialise key_instruct_begin
        key_instruct_begin = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_instruct_begin',
        )
    # create speaker 'read_instruct_begin'
    deviceManager.addDevice(
        deviceName='read_instruct_begin',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # create speaker 'read_thank_you'
    deviceManager.addDevice(
        deviceName='read_thank_you',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "_welcome" ---
    text_welcome = visual.TextStim(win=win, name='text_welcome',
        text='Welcome! This task will take approximately 30 minutes.\n\nBefore we explain the task, we need to first calibrate the eyetracking camera. Please sit in a comfortable position with your head on the chin rest. Once we begin, it is important that you stay in the same position throughout this task.\n\nPlease take a moment to adjust the chair height, chin rest, and sitting posture. Make sure that you feel comfortable and can stay still for a while.\n\n\nWhen you are ready, press any of the white keys to begin',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_welcome = keyboard.Keyboard(deviceName='key_welcome')
    read_welcome = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='read_welcome',    name='read_welcome'
    )
    read_welcome.setVolume(1.0)
    
    # --- Initialize components for Routine "_et_instruct" ---
    text_et = visual.TextStim(win=win, name='text_et',
        text='During the calibration, you will see a target circle moving around the screen. Please try to track it with your eyes.\n\nMake sure to keep looking at the circle when it stops, and follow it when it moves. It is important that you keep your head on the chin rest once this part begins.\n\n\nPress any of the white keys when you are ready, and our team will start the calibration for you',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_et = keyboard.Keyboard(deviceName='key_et')
    read_et = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='read_et',    name='read_et'
    )
    read_et.setVolume(1.0)
    
    # --- Initialize components for Routine "_et_mask" ---
    text_mask = visual.TextStim(win=win, name='text_mask',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "__start__" ---
    text_start = visual.TextStim(win=win, name='text_start',
        text='We are now ready to begin...',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    read_start = sound.Sound(
        'A', 
        secs=1.8, 
        stereo=True, 
        hamming=True, 
        speaker='read_start',    name='read_start'
    )
    read_start.setVolume(1.0)
    # Run 'Begin Experiment' code from trigger_table
    ##TASK ID TRIGGER VALUES##
    # special code 100 (task start, task ID should follow immediately)
    task_start_code = 100
    # special code 102 (task ID for object location WM task)
    task_ID_code = 102
    
    ##GENERAL TRIGGER VALUES##
    # special code 122 (block start)
    block_start_code = 122
    # special code 123 (block end)
    block_end_code = 123
    
    ##TASK SPECIFIC TRIGGER VALUES##
    # N.B.: only use values 1-99 and provide clear comments on used values
    background_start_code = 9
    grid_start_code = 10
    image_1_start_code = 11
    image_2_start_code = 12
    image_3_start_code = 13
    delay_start_code = 14
    prompt_start_code = 15
    test_start_code = 16
    
    # Run 'Begin Experiment' code from task_id
    dev.activate_line(bitmask=task_start_code)  # special code for task start
    core.wait(0.5)  # wait 500ms between two consecutive triggers
    dev.activate_line(bitmask=task_ID_code)  # special code for task ID
    
    etRecord = hardware.eyetracker.EyetrackerControl(
        tracker=eyetracker,
        actionType='Start Only'
    )
    # Run 'Begin Experiment' code from condition_setup
    # Set up condition arrays for the experiment
    rng = np.random.default_rng()
    image_filenames = rng.permutation(['resource/' + str(x).zfill(3) + '.bmp' for x in range(1, 248)])
    image_dot_fn = 'resource/image_dot.png'  # dot image for location trials
    
    conditions = ['objsame', 'objdifferent', 'locsame', 'locdifferent', 'objlocsame', 'objlocdifferent']
    n_conditions = len(conditions)  # 6 different conditions in total
    n_objects_per_trial = 3  # each trial presents 3 objects in a sequence
    locations = [(-0.2, 0.2), (0, 0.2), (0.2, 0.2),
                 (-0.2, 0),             (0.2, 0),
                 (-0.2, -0.2), (0, -0.2), (0.2, -0.2)]
    
    # Practice trials
    n_trials_per_condition = 1  # each condition is presented once during practice
    n_trials_practice = n_conditions * n_trials_per_condition  # 6 trials during practice
    n_objects_practice = n_trials_practice * n_objects_per_trial
    # trial types
    trial_type_practice_list = rng.permutation(conditions * n_trials_per_condition)
    # image locations
    image_loc_practice_list = [rng.permutation(locations) for _ in range(n_trials_practice)]
    # image filenames
    image_fn_practice = image_filenames[:n_objects_practice]
    # package the images into nested lists of 3 objects
    image_fn_practice_list = [image_fn_practice[i * n_objects_per_trial:(i + 1) * n_objects_per_trial] for i in range(n_trials_practice)]
    # only a single novel object used during practice
    image_fn_novel_practice = image_filenames[n_objects_practice]
    
    # Main trials
    n_trials_per_condition = 12  # each condition is repeated 12 times
    n_trials = n_conditions * n_trials_per_condition  # 72 trials during main experiment
    n_objects = n_trials * n_objects_per_trial
    # trial types
    trial_type_list = rng.permutation(conditions * n_trials_per_condition)
    # image locations
    image_loc_list = [rng.permutation(locations) for _ in range(n_trials)]
    # image filenames - skipping images already used for practice trials
    image_fn = image_filenames[n_objects_practice + 1:n_objects_practice + 1 + n_objects]
    # package the images into nested lists of 3 objects
    image_fn_list = [image_fn[i * n_objects_per_trial:(i + 1) * n_objects_per_trial] for i in range(n_trials)]
    # novel objects appear one at a time so no need to package - cast to list for pop()
    image_fn_novel_list = image_filenames[n_objects_practice + 1 + n_objects:].tolist()
    
    assert len(image_filenames) == 247, "Incorrect number of picture stimuli loaded."
    assert len(image_filenames) == (n_objects_per_trial + 1 / n_conditions) * (n_trials_practice + n_trials), "Incorrect number of trials."
    assert np.all(np.array([len(x) for x in image_fn_list]) == n_objects_per_trial), "Incorrect number of stimuli for some trials."
    
    
    # --- Initialize components for Routine "instruct_1" ---
    text_instruct_intro_1 = visual.TextStim(win=win, name='text_instruct_intro_1',
        text='INSTRUCTIONS\n\nIn this experiment you will see a grid with 9 squares. 3 objects will appear one at a time at different locations within the grid. You will be asked to look at these objects, then after a short delay you will be tested on how well you can remember them.\n\n\nPress any of the white keys to continue',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_instruct_intro_1 = keyboard.Keyboard(deviceName='key_instruct_intro_1')
    read_instruct_intro_1 = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='read_instruct_intro_1',    name='read_instruct_intro_1'
    )
    read_instruct_intro_1.setVolume(1.0)
    
    # --- Initialize components for Routine "instruct_2" ---
    text_instruct_intro_2 = visual.TextStim(win=win, name='text_instruct_intro_2',
        text="INSTRUCTIONS\n\nThere will be three different types of trials for this experiment.\n\n1) 'Remember Object' trial. \n2) 'Remember Location' trial. \n3) 'Remember Object and Location' trial. \n\nWe will now explain each type of trial separately. You will see flowcharts of the different trial types, and afterwards you will have a chance to do some practice.\n\n\nPress any of the white keys to continue",
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_instruct_intro_2 = keyboard.Keyboard(deviceName='key_instruct_intro_2')
    read_instruct_intro_2 = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='read_instruct_intro_2',    name='read_instruct_intro_2'
    )
    read_instruct_intro_2.setVolume(1.0)
    
    # --- Initialize components for Routine "instruct_condition" ---
    text_instruct_condition = visual.TextStim(win=win, name='text_instruct_condition',
        text='',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_instruct_condition = keyboard.Keyboard(deviceName='key_instruct_condition')
    read_instruct_condition = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='read_instruct_condition',    name='read_instruct_condition'
    )
    read_instruct_condition.setVolume(1.0)
    
    # --- Initialize components for Routine "diagram_condition" ---
    text_response = visual.TextStim(win=win, name='text_response',
        text='',
        font='Arial',
        units='norm', pos=(0, 0.5), draggable=False, height=0.08, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    image_diagram = visual.ImageStim(
        win=win,
        name='image_diagram', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.2), draggable=False, size=(1.142857, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=False, depth=-2.0)
    text_continue = visual.TextStim(win=win, name='text_continue',
        text='Press any of the white keys to continue',
        font='Arial',
        units='norm', pos=(0, -0.85), draggable=False, height=0.08, wrapWidth=1.8, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    key_diagram_condition = keyboard.Keyboard(deviceName='key_diagram_condition')
    read_instruct_response = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='read_instruct_response',    name='read_instruct_response'
    )
    read_instruct_response.setVolume(1.0)
    
    # --- Initialize components for Routine "instruct_review" ---
    text_instruct_review = visual.TextStim(win=win, name='text_instruct_review',
        text="REVIEW\n\nThere are 3 types of trials in this experiment:\n'Remember Object'\n'Remember Location'\n'Remember Object and Location'.\n\nNote that these 3 types of trials will be intermixed throughout the experiment. This means that you will not know whether you need to respond to the Object or the Location or both until you see the prompt screen. During the delay, try to keep looking at the center of the screen. Please respond as quickly and accurately as possible.\n\n\nPress the green key to start practice trials",
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_instruct_review = keyboard.Keyboard(deviceName='key_instruct_review')
    read_instruct_review = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='read_instruct_review',    name='read_instruct_review'
    )
    read_instruct_review.setVolume(1.0)
    
    # --- Initialize components for Routine "instruct_practice_repeat" ---
    test_practice_repeat = visual.TextStim(win=win, name='test_practice_repeat',
        text="We will repeat the practice trials one more time.\n\nRemember: There are 3 types of trials in this experiment: 'Remember Object', 'Remember Location', and 'Remember Object and Location'.\n\nPress the Green key to indicate that you recognize the object, location, or object in the same location depending on the type of trial, and press the Red key if you don't.\n\nPlease respond as quickly and accurately as possible.\n\n\nPress the green key to start practice trials",
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_practice_repeat = keyboard.Keyboard(deviceName='key_practice_repeat')
    read_practice_repeat = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='read_practice_repeat',    name='read_practice_repeat'
    )
    read_practice_repeat.setVolume(1.0)
    
    # --- Initialize components for Routine "practice_setup" ---
    
    # --- Initialize components for Routine "trial" ---
    background = visual.Rect(
        win=win, name='background',
        width=(1, 1)[0], height=(1, 1)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    grid_outer = visual.Rect(
        win=win, name='grid_outer',
        width=(0.6, 0.6)[0], height=(0.6, 0.6)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    grid_horizontal = visual.Rect(
        win=win, name='grid_horizontal',
        width=(0.6, 0.2)[0], height=(0.6, 0.2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[1.0000, 1.0000, 1.0000],
        opacity=None, depth=-2.0, interpolate=True)
    grid_vertical = visual.Rect(
        win=win, name='grid_vertical',
        width=(0.2, 0.6)[0], height=(0.2, 0.6)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[1.0000, 1.0000, 1.0000],
        opacity=None, depth=-3.0, interpolate=True)
    grid_center = visual.Rect(
        win=win, name='grid_center',
        width=(0.2, 0.2)[0], height=(0.2, 0.2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[1.0000, 1.0000, 1.0000],
        opacity=None, depth=-4.0, interpolate=True)
    image_1 = visual.ImageStim(
        win=win,
        name='image_1', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=False, depth=-5.0)
    image_2 = visual.ImageStim(
        win=win,
        name='image_2', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=False, depth=-6.0)
    image_3 = visual.ImageStim(
        win=win,
        name='image_3', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=False, depth=-7.0)
    text_fixation = visual.TextStim(win=win, name='text_fixation',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    text_prompt = visual.TextStim(win=win, name='text_prompt',
        text='',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-9.0);
    grid_outer_test = visual.Rect(
        win=win, name='grid_outer_test',
        width=(0.6, 0.6)[0], height=(0.6, 0.6)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor='white',
        opacity=None, depth=-10.0, interpolate=True)
    grid_horizontal_test = visual.Rect(
        win=win, name='grid_horizontal_test',
        width=(0.6, 0.2)[0], height=(0.6, 0.2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[1.0000, 1.0000, 1.0000],
        opacity=None, depth=-11.0, interpolate=True)
    grid_vertical_test = visual.Rect(
        win=win, name='grid_vertical_test',
        width=(0.2, 0.6)[0], height=(0.2, 0.6)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[1.0000, 1.0000, 1.0000],
        opacity=None, depth=-12.0, interpolate=True)
    grid_center_test = visual.Rect(
        win=win, name='grid_center_test',
        width=(0.2, 0.2)[0], height=(0.2, 0.2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[1.0000, 1.0000, 1.0000],
        opacity=None, depth=-13.0, interpolate=True)
    image_test = visual.ImageStim(
        win=win,
        name='image_test', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=False, depth=-14.0)
    key_response_test = keyboard.Keyboard(deviceName='key_response_test')
    # Run 'Begin Experiment' code from adjust_image_size
    def scale_to_size(image_object, max_size):
        image_object.size *= max_size / max(image_object.size)  # scale to max size
        image_object._requestedSize = None  # reset for next image original size
    
    
    # --- Initialize components for Routine "practice_feedback" ---
    background_feedback = visual.Rect(
        win=win, name='background_feedback',
        width=(1, 1)[0], height=(1, 1)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    text_feedback = visual.TextStim(win=win, name='text_feedback',
        text='',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "practice_checkpoint" ---
    text_checkpoint = visual.TextStim(win=win, name='text_checkpoint',
        text='Please give us a moment to check whether practice trials need to be repeated...',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_checkpoint = keyboard.Keyboard(deviceName='key_checkpoint')
    read_checkpoint = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='read_checkpoint',    name='read_checkpoint'
    )
    read_checkpoint.setVolume(1.0)
    
    # --- Initialize components for Routine "instruct_begin" ---
    text_instruct_begin = visual.TextStim(win=win, name='text_instruct_begin',
        text='Great job! We will now begin the experiment.\n\nAs a reminder, the trials will ask you to remember the object, the location, or the object and its location. Please keep looking at the center of the screen during the delay.\n\nPress the Green and Red keys accordingly depending on whether you think the object, location, or the object and its location are the same as shown in the 3-object sequence each time. Please respond as quickly and accurately as possible.\n\nNote that you will no longer receive feedback on your responses.\n\n\nPress the green key to start the trials',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_instruct_begin = keyboard.Keyboard(deviceName='key_instruct_begin')
    read_instruct_begin = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='read_instruct_begin',    name='read_instruct_begin'
    )
    read_instruct_begin.setVolume(1.0)
    
    # --- Initialize components for Routine "trial_setup" ---
    
    # --- Initialize components for Routine "trial" ---
    background = visual.Rect(
        win=win, name='background',
        width=(1, 1)[0], height=(1, 1)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    grid_outer = visual.Rect(
        win=win, name='grid_outer',
        width=(0.6, 0.6)[0], height=(0.6, 0.6)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    grid_horizontal = visual.Rect(
        win=win, name='grid_horizontal',
        width=(0.6, 0.2)[0], height=(0.6, 0.2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[1.0000, 1.0000, 1.0000],
        opacity=None, depth=-2.0, interpolate=True)
    grid_vertical = visual.Rect(
        win=win, name='grid_vertical',
        width=(0.2, 0.6)[0], height=(0.2, 0.6)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[1.0000, 1.0000, 1.0000],
        opacity=None, depth=-3.0, interpolate=True)
    grid_center = visual.Rect(
        win=win, name='grid_center',
        width=(0.2, 0.2)[0], height=(0.2, 0.2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[1.0000, 1.0000, 1.0000],
        opacity=None, depth=-4.0, interpolate=True)
    image_1 = visual.ImageStim(
        win=win,
        name='image_1', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=False, depth=-5.0)
    image_2 = visual.ImageStim(
        win=win,
        name='image_2', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=False, depth=-6.0)
    image_3 = visual.ImageStim(
        win=win,
        name='image_3', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=False, depth=-7.0)
    text_fixation = visual.TextStim(win=win, name='text_fixation',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    text_prompt = visual.TextStim(win=win, name='text_prompt',
        text='',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-9.0);
    grid_outer_test = visual.Rect(
        win=win, name='grid_outer_test',
        width=(0.6, 0.6)[0], height=(0.6, 0.6)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor='white',
        opacity=None, depth=-10.0, interpolate=True)
    grid_horizontal_test = visual.Rect(
        win=win, name='grid_horizontal_test',
        width=(0.6, 0.2)[0], height=(0.6, 0.2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[1.0000, 1.0000, 1.0000],
        opacity=None, depth=-11.0, interpolate=True)
    grid_vertical_test = visual.Rect(
        win=win, name='grid_vertical_test',
        width=(0.2, 0.6)[0], height=(0.2, 0.6)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[1.0000, 1.0000, 1.0000],
        opacity=None, depth=-12.0, interpolate=True)
    grid_center_test = visual.Rect(
        win=win, name='grid_center_test',
        width=(0.2, 0.2)[0], height=(0.2, 0.2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[1.0000, 1.0000, 1.0000],
        opacity=None, depth=-13.0, interpolate=True)
    image_test = visual.ImageStim(
        win=win,
        name='image_test', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=False, depth=-14.0)
    key_response_test = keyboard.Keyboard(deviceName='key_response_test')
    # Run 'Begin Experiment' code from adjust_image_size
    def scale_to_size(image_object, max_size):
        image_object.size *= max_size / max(image_object.size)  # scale to max size
        image_object._requestedSize = None  # reset for next image original size
    
    
    # --- Initialize components for Routine "__end__" ---
    text_thank_you = visual.TextStim(win=win, name='text_thank_you',
        text='Thank you. You have completed this task!',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    read_thank_you = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='read_thank_you',    name='read_thank_you'
    )
    read_thank_you.setVolume(1.0)
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "_welcome" ---
    # create an object to store info about Routine _welcome
    _welcome = data.Routine(
        name='_welcome',
        components=[text_welcome, key_welcome, read_welcome],
    )
    _welcome.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_welcome
    key_welcome.keys = []
    key_welcome.rt = []
    _key_welcome_allKeys = []
    read_welcome.setSound('resource/welcome.wav', hamming=True)
    read_welcome.setVolume(1.0, log=False)
    read_welcome.seek(0)
    # store start times for _welcome
    _welcome.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    _welcome.tStart = globalClock.getTime(format='float')
    _welcome.status = STARTED
    _welcome.maxDuration = None
    # keep track of which components have finished
    _welcomeComponents = _welcome.components
    for thisComponent in _welcome.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "_welcome" ---
    _welcome.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_welcome* updates
        
        # if text_welcome is starting this frame...
        if text_welcome.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_welcome.frameNStart = frameN  # exact frame index
            text_welcome.tStart = t  # local t and not account for scr refresh
            text_welcome.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_welcome, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_welcome.status = STARTED
            text_welcome.setAutoDraw(True)
        
        # if text_welcome is active this frame...
        if text_welcome.status == STARTED:
            # update params
            pass
        
        # *key_welcome* updates
        waitOnFlip = False
        
        # if key_welcome is starting this frame...
        if key_welcome.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
            # keep track of start time/frame for later
            key_welcome.frameNStart = frameN  # exact frame index
            key_welcome.tStart = t  # local t and not account for scr refresh
            key_welcome.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_welcome, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_welcome.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_welcome.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_welcome.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_welcome.status == STARTED and not waitOnFlip:
            theseKeys = key_welcome.getKeys(keyList=['3', '4', '5', '6'], ignoreKeys=["escape"], waitRelease=True)
            _key_welcome_allKeys.extend(theseKeys)
            if len(_key_welcome_allKeys):
                key_welcome.keys = _key_welcome_allKeys[-1].name  # just the last key pressed
                key_welcome.rt = _key_welcome_allKeys[-1].rt
                key_welcome.duration = _key_welcome_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *read_welcome* updates
        
        # if read_welcome is starting this frame...
        if read_welcome.status == NOT_STARTED and tThisFlip >= 0.8-frameTolerance:
            # keep track of start time/frame for later
            read_welcome.frameNStart = frameN  # exact frame index
            read_welcome.tStart = t  # local t and not account for scr refresh
            read_welcome.tStartRefresh = tThisFlipGlobal  # on global time
            # update status
            read_welcome.status = STARTED
            read_welcome.play(when=win)  # sync with win flip
        
        # if read_welcome is stopping this frame...
        if read_welcome.status == STARTED:
            if bool(False) or read_welcome.isFinished:
                # keep track of stop time/frame for later
                read_welcome.tStop = t  # not accounting for scr refresh
                read_welcome.tStopRefresh = tThisFlipGlobal  # on global time
                read_welcome.frameNStop = frameN  # exact frame index
                # update status
                read_welcome.status = FINISHED
                read_welcome.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[read_welcome]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            _welcome.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in _welcome.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "_welcome" ---
    for thisComponent in _welcome.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for _welcome
    _welcome.tStop = globalClock.getTime(format='float')
    _welcome.tStopRefresh = tThisFlipGlobal
    read_welcome.pause()  # ensure sound has stopped at end of Routine
    thisExp.nextEntry()
    # the Routine "_welcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "_et_instruct" ---
    # create an object to store info about Routine _et_instruct
    _et_instruct = data.Routine(
        name='_et_instruct',
        components=[text_et, key_et, read_et],
    )
    _et_instruct.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_et
    key_et.keys = []
    key_et.rt = []
    _key_et_allKeys = []
    read_et.setSound('resource/eyetrack_calibrate_instruct.wav', hamming=True)
    read_et.setVolume(1.0, log=False)
    read_et.seek(0)
    # store start times for _et_instruct
    _et_instruct.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    _et_instruct.tStart = globalClock.getTime(format='float')
    _et_instruct.status = STARTED
    _et_instruct.maxDuration = None
    # keep track of which components have finished
    _et_instructComponents = _et_instruct.components
    for thisComponent in _et_instruct.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "_et_instruct" ---
    _et_instruct.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_et* updates
        
        # if text_et is starting this frame...
        if text_et.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_et.frameNStart = frameN  # exact frame index
            text_et.tStart = t  # local t and not account for scr refresh
            text_et.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_et, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_et.status = STARTED
            text_et.setAutoDraw(True)
        
        # if text_et is active this frame...
        if text_et.status == STARTED:
            # update params
            pass
        
        # *key_et* updates
        waitOnFlip = False
        
        # if key_et is starting this frame...
        if key_et.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
            # keep track of start time/frame for later
            key_et.frameNStart = frameN  # exact frame index
            key_et.tStart = t  # local t and not account for scr refresh
            key_et.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_et, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_et.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_et.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_et.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_et.status == STARTED and not waitOnFlip:
            theseKeys = key_et.getKeys(keyList=['3', '4', '5', '6'], ignoreKeys=["escape"], waitRelease=True)
            _key_et_allKeys.extend(theseKeys)
            if len(_key_et_allKeys):
                key_et.keys = _key_et_allKeys[-1].name  # just the last key pressed
                key_et.rt = _key_et_allKeys[-1].rt
                key_et.duration = _key_et_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *read_et* updates
        
        # if read_et is starting this frame...
        if read_et.status == NOT_STARTED and tThisFlip >= 0.8-frameTolerance:
            # keep track of start time/frame for later
            read_et.frameNStart = frameN  # exact frame index
            read_et.tStart = t  # local t and not account for scr refresh
            read_et.tStartRefresh = tThisFlipGlobal  # on global time
            # update status
            read_et.status = STARTED
            read_et.play(when=win)  # sync with win flip
        
        # if read_et is stopping this frame...
        if read_et.status == STARTED:
            if bool(False) or read_et.isFinished:
                # keep track of stop time/frame for later
                read_et.tStop = t  # not accounting for scr refresh
                read_et.tStopRefresh = tThisFlipGlobal  # on global time
                read_et.frameNStop = frameN  # exact frame index
                # update status
                read_et.status = FINISHED
                read_et.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[read_et]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            _et_instruct.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in _et_instruct.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "_et_instruct" ---
    for thisComponent in _et_instruct.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for _et_instruct
    _et_instruct.tStop = globalClock.getTime(format='float')
    _et_instruct.tStopRefresh = tThisFlipGlobal
    read_et.pause()  # ensure sound has stopped at end of Routine
    thisExp.nextEntry()
    # the Routine "_et_instruct" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "_et_mask" ---
    # create an object to store info about Routine _et_mask
    _et_mask = data.Routine(
        name='_et_mask',
        components=[text_mask],
    )
    _et_mask.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for _et_mask
    _et_mask.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    _et_mask.tStart = globalClock.getTime(format='float')
    _et_mask.status = STARTED
    _et_mask.maxDuration = None
    # keep track of which components have finished
    _et_maskComponents = _et_mask.components
    for thisComponent in _et_mask.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "_et_mask" ---
    _et_mask.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 0.05:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_mask* updates
        
        # if text_mask is starting this frame...
        if text_mask.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_mask.frameNStart = frameN  # exact frame index
            text_mask.tStart = t  # local t and not account for scr refresh
            text_mask.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_mask, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_mask.status = STARTED
            text_mask.setAutoDraw(True)
        
        # if text_mask is active this frame...
        if text_mask.status == STARTED:
            # update params
            pass
        
        # if text_mask is stopping this frame...
        if text_mask.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_mask.tStartRefresh + 0.05-frameTolerance:
                # keep track of stop time/frame for later
                text_mask.tStop = t  # not accounting for scr refresh
                text_mask.tStopRefresh = tThisFlipGlobal  # on global time
                text_mask.frameNStop = frameN  # exact frame index
                # update status
                text_mask.status = FINISHED
                text_mask.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            _et_mask.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in _et_mask.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "_et_mask" ---
    for thisComponent in _et_mask.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for _et_mask
    _et_mask.tStop = globalClock.getTime(format='float')
    _et_mask.tStopRefresh = tThisFlipGlobal
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if _et_mask.maxDurationReached:
        routineTimer.addTime(-_et_mask.maxDuration)
    elif _et_mask.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-0.050000)
    thisExp.nextEntry()
    # define target for _et_cal
    _et_calTarget = visual.TargetStim(win, 
        name='_et_calTarget',
        radius=0.015, fillColor='white', borderColor='green', lineWidth=2.0,
        innerRadius=0.005, innerFillColor='black', innerBorderColor='black', innerLineWidth=2.0,
        colorSpace='rgb', units=None
    )
    # define parameters for _et_cal
    _et_cal = hardware.eyetracker.EyetrackerCalibration(win, 
        eyetracker, _et_calTarget,
        units=None, colorSpace='rgb',
        progressMode='time', targetDur=1.5, expandScale=1.5,
        targetLayout='NINE_POINTS', randomisePos=True, textColor='white',
        movementAnimation=True, targetDelay=1.0
    )
    # run calibration
    _et_cal.run()
    # clear any keypresses from during _et_cal so they don't interfere with the experiment
    defaultKeyboard.clearEvents()
    thisExp.nextEntry()
    # the Routine "_et_cal" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "__start__" ---
    # create an object to store info about Routine __start__
    __start__ = data.Routine(
        name='__start__',
        components=[text_start, read_start, etRecord],
    )
    __start__.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    read_start.setSound('resource/ready_to_begin.wav', secs=1.8, hamming=True)
    read_start.setVolume(1.0, log=False)
    read_start.seek(0)
    # store start times for __start__
    __start__.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    __start__.tStart = globalClock.getTime(format='float')
    __start__.status = STARTED
    __start__.maxDuration = None
    # keep track of which components have finished
    __start__Components = __start__.components
    for thisComponent in __start__.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "__start__" ---
    __start__.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 2.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_start* updates
        
        # if text_start is starting this frame...
        if text_start.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_start.frameNStart = frameN  # exact frame index
            text_start.tStart = t  # local t and not account for scr refresh
            text_start.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_start, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_start.status = STARTED
            text_start.setAutoDraw(True)
        
        # if text_start is active this frame...
        if text_start.status == STARTED:
            # update params
            pass
        
        # if text_start is stopping this frame...
        if text_start.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_start.tStartRefresh + 2.0-frameTolerance:
                # keep track of stop time/frame for later
                text_start.tStop = t  # not accounting for scr refresh
                text_start.tStopRefresh = tThisFlipGlobal  # on global time
                text_start.frameNStop = frameN  # exact frame index
                # update status
                text_start.status = FINISHED
                text_start.setAutoDraw(False)
        
        # *read_start* updates
        
        # if read_start is starting this frame...
        if read_start.status == NOT_STARTED and t >= 0.2-frameTolerance:
            # keep track of start time/frame for later
            read_start.frameNStart = frameN  # exact frame index
            read_start.tStart = t  # local t and not account for scr refresh
            read_start.tStartRefresh = tThisFlipGlobal  # on global time
            # update status
            read_start.status = STARTED
            read_start.play()  # start the sound (it finishes automatically)
        
        # if read_start is stopping this frame...
        if read_start.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > read_start.tStartRefresh + 1.8-frameTolerance or read_start.isFinished:
                # keep track of stop time/frame for later
                read_start.tStop = t  # not accounting for scr refresh
                read_start.tStopRefresh = tThisFlipGlobal  # on global time
                read_start.frameNStop = frameN  # exact frame index
                # update status
                read_start.status = FINISHED
                read_start.stop()
        
        # *etRecord* updates
        
        # if etRecord is starting this frame...
        if etRecord.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            etRecord.frameNStart = frameN  # exact frame index
            etRecord.tStart = t  # local t and not account for scr refresh
            etRecord.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(etRecord, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('etRecord.started', t)
            # update status
            etRecord.status = STARTED
            etRecord.start()
        if etRecord.status == STARTED:
            etRecord.tStop = t  # not accounting for scr refresh
            etRecord.tStopRefresh = tThisFlipGlobal  # on global time
            etRecord.frameNStop = frameN  # exact frame index
            etRecord.status = FINISHED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[read_start]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            __start__.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in __start__.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "__start__" ---
    for thisComponent in __start__.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for __start__
    __start__.tStop = globalClock.getTime(format='float')
    __start__.tStopRefresh = tThisFlipGlobal
    read_start.pause()  # ensure sound has stopped at end of Routine
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if __start__.maxDurationReached:
        routineTimer.addTime(-__start__.maxDuration)
    elif __start__.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-2.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "instruct_1" ---
    # create an object to store info about Routine instruct_1
    instruct_1 = data.Routine(
        name='instruct_1',
        components=[text_instruct_intro_1, key_instruct_intro_1, read_instruct_intro_1],
    )
    instruct_1.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_instruct_intro_1
    key_instruct_intro_1.keys = []
    key_instruct_intro_1.rt = []
    _key_instruct_intro_1_allKeys = []
    read_instruct_intro_1.setSound('resource/instruct_1.wav', hamming=True)
    read_instruct_intro_1.setVolume(1.0, log=False)
    read_instruct_intro_1.seek(0)
    # store start times for instruct_1
    instruct_1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instruct_1.tStart = globalClock.getTime(format='float')
    instruct_1.status = STARTED
    instruct_1.maxDuration = None
    # keep track of which components have finished
    instruct_1Components = instruct_1.components
    for thisComponent in instruct_1.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instruct_1" ---
    instruct_1.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_instruct_intro_1* updates
        
        # if text_instruct_intro_1 is starting this frame...
        if text_instruct_intro_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_instruct_intro_1.frameNStart = frameN  # exact frame index
            text_instruct_intro_1.tStart = t  # local t and not account for scr refresh
            text_instruct_intro_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_instruct_intro_1, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_instruct_intro_1.status = STARTED
            text_instruct_intro_1.setAutoDraw(True)
        
        # if text_instruct_intro_1 is active this frame...
        if text_instruct_intro_1.status == STARTED:
            # update params
            pass
        
        # *key_instruct_intro_1* updates
        waitOnFlip = False
        
        # if key_instruct_intro_1 is starting this frame...
        if key_instruct_intro_1.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
            # keep track of start time/frame for later
            key_instruct_intro_1.frameNStart = frameN  # exact frame index
            key_instruct_intro_1.tStart = t  # local t and not account for scr refresh
            key_instruct_intro_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_instruct_intro_1, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_instruct_intro_1.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_instruct_intro_1.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_instruct_intro_1.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_instruct_intro_1.status == STARTED and not waitOnFlip:
            theseKeys = key_instruct_intro_1.getKeys(keyList=['3', '4', '5', '6'], ignoreKeys=["escape"], waitRelease=True)
            _key_instruct_intro_1_allKeys.extend(theseKeys)
            if len(_key_instruct_intro_1_allKeys):
                key_instruct_intro_1.keys = _key_instruct_intro_1_allKeys[-1].name  # just the last key pressed
                key_instruct_intro_1.rt = _key_instruct_intro_1_allKeys[-1].rt
                key_instruct_intro_1.duration = _key_instruct_intro_1_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *read_instruct_intro_1* updates
        
        # if read_instruct_intro_1 is starting this frame...
        if read_instruct_intro_1.status == NOT_STARTED and tThisFlip >= 0.8-frameTolerance:
            # keep track of start time/frame for later
            read_instruct_intro_1.frameNStart = frameN  # exact frame index
            read_instruct_intro_1.tStart = t  # local t and not account for scr refresh
            read_instruct_intro_1.tStartRefresh = tThisFlipGlobal  # on global time
            # update status
            read_instruct_intro_1.status = STARTED
            read_instruct_intro_1.play(when=win)  # sync with win flip
        
        # if read_instruct_intro_1 is stopping this frame...
        if read_instruct_intro_1.status == STARTED:
            if bool(False) or read_instruct_intro_1.isFinished:
                # keep track of stop time/frame for later
                read_instruct_intro_1.tStop = t  # not accounting for scr refresh
                read_instruct_intro_1.tStopRefresh = tThisFlipGlobal  # on global time
                read_instruct_intro_1.frameNStop = frameN  # exact frame index
                # update status
                read_instruct_intro_1.status = FINISHED
                read_instruct_intro_1.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[read_instruct_intro_1]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instruct_1.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instruct_1.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruct_1" ---
    for thisComponent in instruct_1.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instruct_1
    instruct_1.tStop = globalClock.getTime(format='float')
    instruct_1.tStopRefresh = tThisFlipGlobal
    read_instruct_intro_1.pause()  # ensure sound has stopped at end of Routine
    thisExp.nextEntry()
    # the Routine "instruct_1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instruct_2" ---
    # create an object to store info about Routine instruct_2
    instruct_2 = data.Routine(
        name='instruct_2',
        components=[text_instruct_intro_2, key_instruct_intro_2, read_instruct_intro_2],
    )
    instruct_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_instruct_intro_2
    key_instruct_intro_2.keys = []
    key_instruct_intro_2.rt = []
    _key_instruct_intro_2_allKeys = []
    read_instruct_intro_2.setSound('resource/instruct_2.wav', hamming=True)
    read_instruct_intro_2.setVolume(1.0, log=False)
    read_instruct_intro_2.seek(0)
    # store start times for instruct_2
    instruct_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instruct_2.tStart = globalClock.getTime(format='float')
    instruct_2.status = STARTED
    instruct_2.maxDuration = None
    # keep track of which components have finished
    instruct_2Components = instruct_2.components
    for thisComponent in instruct_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instruct_2" ---
    instruct_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_instruct_intro_2* updates
        
        # if text_instruct_intro_2 is starting this frame...
        if text_instruct_intro_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_instruct_intro_2.frameNStart = frameN  # exact frame index
            text_instruct_intro_2.tStart = t  # local t and not account for scr refresh
            text_instruct_intro_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_instruct_intro_2, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_instruct_intro_2.status = STARTED
            text_instruct_intro_2.setAutoDraw(True)
        
        # if text_instruct_intro_2 is active this frame...
        if text_instruct_intro_2.status == STARTED:
            # update params
            pass
        
        # *key_instruct_intro_2* updates
        waitOnFlip = False
        
        # if key_instruct_intro_2 is starting this frame...
        if key_instruct_intro_2.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
            # keep track of start time/frame for later
            key_instruct_intro_2.frameNStart = frameN  # exact frame index
            key_instruct_intro_2.tStart = t  # local t and not account for scr refresh
            key_instruct_intro_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_instruct_intro_2, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_instruct_intro_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_instruct_intro_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_instruct_intro_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_instruct_intro_2.status == STARTED and not waitOnFlip:
            theseKeys = key_instruct_intro_2.getKeys(keyList=['3', '4', '5', '6'], ignoreKeys=["escape"], waitRelease=True)
            _key_instruct_intro_2_allKeys.extend(theseKeys)
            if len(_key_instruct_intro_2_allKeys):
                key_instruct_intro_2.keys = _key_instruct_intro_2_allKeys[-1].name  # just the last key pressed
                key_instruct_intro_2.rt = _key_instruct_intro_2_allKeys[-1].rt
                key_instruct_intro_2.duration = _key_instruct_intro_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *read_instruct_intro_2* updates
        
        # if read_instruct_intro_2 is starting this frame...
        if read_instruct_intro_2.status == NOT_STARTED and tThisFlip >= 0.8-frameTolerance:
            # keep track of start time/frame for later
            read_instruct_intro_2.frameNStart = frameN  # exact frame index
            read_instruct_intro_2.tStart = t  # local t and not account for scr refresh
            read_instruct_intro_2.tStartRefresh = tThisFlipGlobal  # on global time
            # update status
            read_instruct_intro_2.status = STARTED
            read_instruct_intro_2.play(when=win)  # sync with win flip
        
        # if read_instruct_intro_2 is stopping this frame...
        if read_instruct_intro_2.status == STARTED:
            if bool(False) or read_instruct_intro_2.isFinished:
                # keep track of stop time/frame for later
                read_instruct_intro_2.tStop = t  # not accounting for scr refresh
                read_instruct_intro_2.tStopRefresh = tThisFlipGlobal  # on global time
                read_instruct_intro_2.frameNStop = frameN  # exact frame index
                # update status
                read_instruct_intro_2.status = FINISHED
                read_instruct_intro_2.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[read_instruct_intro_2]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instruct_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instruct_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruct_2" ---
    for thisComponent in instruct_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instruct_2
    instruct_2.tStop = globalClock.getTime(format='float')
    instruct_2.tStopRefresh = tThisFlipGlobal
    read_instruct_intro_2.pause()  # ensure sound has stopped at end of Routine
    thisExp.nextEntry()
    # the Routine "instruct_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    instruct_loop = data.TrialHandler2(
        name='instruct_loop',
        nReps=3.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(instruct_loop)  # add the loop to the experiment
    thisInstruct_loop = instruct_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisInstruct_loop.rgb)
    if thisInstruct_loop != None:
        for paramName in thisInstruct_loop:
            globals()[paramName] = thisInstruct_loop[paramName]
    
    for thisInstruct_loop in instruct_loop:
        currentLoop = instruct_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisInstruct_loop.rgb)
        if thisInstruct_loop != None:
            for paramName in thisInstruct_loop:
                globals()[paramName] = thisInstruct_loop[paramName]
        
        # --- Prepare to start Routine "instruct_condition" ---
        # create an object to store info about Routine instruct_condition
        instruct_condition = data.Routine(
            name='instruct_condition',
            components=[text_instruct_condition, key_instruct_condition, read_instruct_condition],
        )
        instruct_condition.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from set_instruct_content
        # Remember Object Condition
        if instruct_loop.thisRepN == 0:
            instruct_condition_text = "Remember Object trial: In this type of trial you need to remember the identity of the objects shown to you.\n"
            instruct_condition_text += "\n"
            instruct_condition_text += "You will see 3 objects appearing one at a time followed by an 8-second delay."
            instruct_condition_text += "You will then be shown a prompt screen saying 'Remember Object'."
            instruct_condition_text += "This prompt screen will be followed by a test object in the center of the grid.\n"
            instruct_condition_text += "\n"
            instruct_condition_text += "'Remember Object' tells you that you need to decide whether this test object was one of the 3 objects just shown to you.\n"
            instruct_condition_text += "\n\n"
            instruct_condition_text += "Press any of the white keys to continue"
        
            instruct_condition_audio_filename = "resource/instruct_condition_object.wav"
            instruct_response_audio_filename = "resource/instruct_response_object.wav"
            instruct_diagram_filename = "resource/object_trial_diagram.tif"
        
            instruct_response_text = "Remember Object Trial:\n"
            instruct_response_text += "\n"
            instruct_response_text += "If you recognize the test object, press the Green key to indicate that YES, the test object was in the 3-object sequence.\n"
            instruct_response_text += "\n"
            instruct_response_text += "If you do NOT recognize the test object, press the Red key to indicate that NO, the test object was NOT in the 3-object sequence."
        
        # Remember Location Condition
        elif instruct_loop.thisRepN == 1:
            instruct_condition_text = "Remember Location trial: In this type of trial you need to remember the location of the objects shown to you.\n"
            instruct_condition_text += "\n"
            instruct_condition_text += "You will see 3 objects appearing one at a time followed by an 8-second delay."
            instruct_condition_text += "You will then be shown a prompt screen saying 'Remember Location'."
            instruct_condition_text += "This prompt screen will be followed by a dot in one of the squares of the grid.\n"
            instruct_condition_text += "\n"
            instruct_condition_text += "'Remember Location' tells you that you need to decide whether this square was previously occupied by any object in the 3-object sequence.\n"
            instruct_condition_text += "\n\n"
            instruct_condition_text += "Press any of the white keys to continue"
        
            instruct_condition_audio_filename = "resource/instruct_condition_location.wav"
            instruct_response_audio_filename = "resource/instruct_response_location.wav"
            instruct_diagram_filename = "resource/location_trial_diagram.tif"
        
            instruct_response_text = "Remember Location Trial:\n"
            instruct_response_text += "\n"
            instruct_response_text += "If the square indicated by the dot was previously occupied, press the Green key to indicate that YES, one of the objects in the 3-object sequence was in this square.\n"
            instruct_response_text += "\n"
            instruct_response_text += "If the square indicated by the dot was NOT previously occupied, press the Red key to indicate that NO, none of the objects in the 3-object sequence were in this square."    
        
        # Remember Object and Location Condition
        elif instruct_loop.thisRepN == 2:
            instruct_condition_text = "Remember Object and Location trial: In this type of trial you need to remember both the identity of the objects and their locations.\n"
            instruct_condition_text += "\n"
            instruct_condition_text += "As with the other trials, you will see 3 objects appearing one at a time followed by an 8-second delay."
            instruct_condition_text += "You will then be shown a prompt screen saying 'Remember Object and Location'."
            instruct_condition_text += "This prompt screen will be followed by a test object in one of the squares of the grid.\n"
            instruct_condition_text += "\n"
            instruct_condition_text += "'Remember Object and Location' tells you that you need to decide whether this object is in the same location. In other words, whether the test object is in the same square as it was in the 3-object sequence.\n"
            instruct_condition_text += "\n\n"
            instruct_condition_text += "Press any of the white keys to continue"
        
            instruct_condition_audio_filename = "resource/instruct_condition_object_and_location.wav"
            instruct_response_audio_filename = "resource/instruct_response_object_and_location.wav"
            instruct_diagram_filename = "resource/object_location_trial_diagram.tif"
        
            instruct_response_text = "Remember Object and Location Trial:\n"
            instruct_response_text += "\n"
            instruct_response_text += "If the test object is in the same location, press the Green key to indicate that YES, the test object is in the same square as it was in the 3-object sequence.\n"
            instruct_response_text += "\n"
            instruct_response_text += "If the test object is NOT in the same location, press the Red key to indicate that NO, the test object is NOT in the same square as it was in the 3-object sequence."
        
        text_instruct_condition.setText(instruct_condition_text)
        # create starting attributes for key_instruct_condition
        key_instruct_condition.keys = []
        key_instruct_condition.rt = []
        _key_instruct_condition_allKeys = []
        read_instruct_condition.setSound(instruct_condition_audio_filename, hamming=True)
        read_instruct_condition.setVolume(1.0, log=False)
        read_instruct_condition.seek(0)
        # store start times for instruct_condition
        instruct_condition.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        instruct_condition.tStart = globalClock.getTime(format='float')
        instruct_condition.status = STARTED
        instruct_condition.maxDuration = None
        # keep track of which components have finished
        instruct_conditionComponents = instruct_condition.components
        for thisComponent in instruct_condition.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "instruct_condition" ---
        # if trial has changed, end Routine now
        if isinstance(instruct_loop, data.TrialHandler2) and thisInstruct_loop.thisN != instruct_loop.thisTrial.thisN:
            continueRoutine = False
        instruct_condition.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_instruct_condition* updates
            
            # if text_instruct_condition is starting this frame...
            if text_instruct_condition.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_instruct_condition.frameNStart = frameN  # exact frame index
                text_instruct_condition.tStart = t  # local t and not account for scr refresh
                text_instruct_condition.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_instruct_condition, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_instruct_condition.status = STARTED
                text_instruct_condition.setAutoDraw(True)
            
            # if text_instruct_condition is active this frame...
            if text_instruct_condition.status == STARTED:
                # update params
                pass
            
            # *key_instruct_condition* updates
            waitOnFlip = False
            
            # if key_instruct_condition is starting this frame...
            if key_instruct_condition.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
                # keep track of start time/frame for later
                key_instruct_condition.frameNStart = frameN  # exact frame index
                key_instruct_condition.tStart = t  # local t and not account for scr refresh
                key_instruct_condition.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_instruct_condition, 'tStartRefresh')  # time at next scr refresh
                # update status
                key_instruct_condition.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_instruct_condition.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_instruct_condition.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_instruct_condition.status == STARTED and not waitOnFlip:
                theseKeys = key_instruct_condition.getKeys(keyList=['3', '4', '5', '6'], ignoreKeys=["escape"], waitRelease=True)
                _key_instruct_condition_allKeys.extend(theseKeys)
                if len(_key_instruct_condition_allKeys):
                    key_instruct_condition.keys = _key_instruct_condition_allKeys[-1].name  # just the last key pressed
                    key_instruct_condition.rt = _key_instruct_condition_allKeys[-1].rt
                    key_instruct_condition.duration = _key_instruct_condition_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *read_instruct_condition* updates
            
            # if read_instruct_condition is starting this frame...
            if read_instruct_condition.status == NOT_STARTED and tThisFlip >= 0.8-frameTolerance:
                # keep track of start time/frame for later
                read_instruct_condition.frameNStart = frameN  # exact frame index
                read_instruct_condition.tStart = t  # local t and not account for scr refresh
                read_instruct_condition.tStartRefresh = tThisFlipGlobal  # on global time
                # update status
                read_instruct_condition.status = STARTED
                read_instruct_condition.play(when=win)  # sync with win flip
            
            # if read_instruct_condition is stopping this frame...
            if read_instruct_condition.status == STARTED:
                if bool(False) or read_instruct_condition.isFinished:
                    # keep track of stop time/frame for later
                    read_instruct_condition.tStop = t  # not accounting for scr refresh
                    read_instruct_condition.tStopRefresh = tThisFlipGlobal  # on global time
                    read_instruct_condition.frameNStop = frameN  # exact frame index
                    # update status
                    read_instruct_condition.status = FINISHED
                    read_instruct_condition.stop()
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[read_instruct_condition]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                instruct_condition.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in instruct_condition.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "instruct_condition" ---
        for thisComponent in instruct_condition.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for instruct_condition
        instruct_condition.tStop = globalClock.getTime(format='float')
        instruct_condition.tStopRefresh = tThisFlipGlobal
        read_instruct_condition.pause()  # ensure sound has stopped at end of Routine
        # the Routine "instruct_condition" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "diagram_condition" ---
        # create an object to store info about Routine diagram_condition
        diagram_condition = data.Routine(
            name='diagram_condition',
            components=[text_response, image_diagram, text_continue, key_diagram_condition, read_instruct_response],
        )
        diagram_condition.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        text_response.setText(instruct_response_text)
        # Run 'Begin Routine' code from align_text
        text_response.alignText = 'left'  # align text to the left
        
        image_diagram.setImage(instruct_diagram_filename)
        # create starting attributes for key_diagram_condition
        key_diagram_condition.keys = []
        key_diagram_condition.rt = []
        _key_diagram_condition_allKeys = []
        read_instruct_response.setSound(instruct_response_audio_filename, hamming=True)
        read_instruct_response.setVolume(1.0, log=False)
        read_instruct_response.seek(0)
        # store start times for diagram_condition
        diagram_condition.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        diagram_condition.tStart = globalClock.getTime(format='float')
        diagram_condition.status = STARTED
        diagram_condition.maxDuration = None
        # keep track of which components have finished
        diagram_conditionComponents = diagram_condition.components
        for thisComponent in diagram_condition.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "diagram_condition" ---
        # if trial has changed, end Routine now
        if isinstance(instruct_loop, data.TrialHandler2) and thisInstruct_loop.thisN != instruct_loop.thisTrial.thisN:
            continueRoutine = False
        diagram_condition.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_response* updates
            
            # if text_response is starting this frame...
            if text_response.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_response.frameNStart = frameN  # exact frame index
                text_response.tStart = t  # local t and not account for scr refresh
                text_response.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_response, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_response.status = STARTED
                text_response.setAutoDraw(True)
            
            # if text_response is active this frame...
            if text_response.status == STARTED:
                # update params
                pass
            
            # *image_diagram* updates
            
            # if image_diagram is starting this frame...
            if image_diagram.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_diagram.frameNStart = frameN  # exact frame index
                image_diagram.tStart = t  # local t and not account for scr refresh
                image_diagram.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_diagram, 'tStartRefresh')  # time at next scr refresh
                # update status
                image_diagram.status = STARTED
                image_diagram.setAutoDraw(True)
            
            # if image_diagram is active this frame...
            if image_diagram.status == STARTED:
                # update params
                pass
            
            # *text_continue* updates
            
            # if text_continue is starting this frame...
            if text_continue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_continue.frameNStart = frameN  # exact frame index
                text_continue.tStart = t  # local t and not account for scr refresh
                text_continue.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_continue, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_continue.status = STARTED
                text_continue.setAutoDraw(True)
            
            # if text_continue is active this frame...
            if text_continue.status == STARTED:
                # update params
                pass
            
            # *key_diagram_condition* updates
            waitOnFlip = False
            
            # if key_diagram_condition is starting this frame...
            if key_diagram_condition.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
                # keep track of start time/frame for later
                key_diagram_condition.frameNStart = frameN  # exact frame index
                key_diagram_condition.tStart = t  # local t and not account for scr refresh
                key_diagram_condition.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_diagram_condition, 'tStartRefresh')  # time at next scr refresh
                # update status
                key_diagram_condition.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_diagram_condition.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_diagram_condition.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_diagram_condition.status == STARTED and not waitOnFlip:
                theseKeys = key_diagram_condition.getKeys(keyList=['3', '4', '5', '6'], ignoreKeys=["escape"], waitRelease=True)
                _key_diagram_condition_allKeys.extend(theseKeys)
                if len(_key_diagram_condition_allKeys):
                    key_diagram_condition.keys = _key_diagram_condition_allKeys[-1].name  # just the last key pressed
                    key_diagram_condition.rt = _key_diagram_condition_allKeys[-1].rt
                    key_diagram_condition.duration = _key_diagram_condition_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *read_instruct_response* updates
            
            # if read_instruct_response is starting this frame...
            if read_instruct_response.status == NOT_STARTED and tThisFlip >= 0.8-frameTolerance:
                # keep track of start time/frame for later
                read_instruct_response.frameNStart = frameN  # exact frame index
                read_instruct_response.tStart = t  # local t and not account for scr refresh
                read_instruct_response.tStartRefresh = tThisFlipGlobal  # on global time
                # update status
                read_instruct_response.status = STARTED
                read_instruct_response.play(when=win)  # sync with win flip
            
            # if read_instruct_response is stopping this frame...
            if read_instruct_response.status == STARTED:
                if bool(False) or read_instruct_response.isFinished:
                    # keep track of stop time/frame for later
                    read_instruct_response.tStop = t  # not accounting for scr refresh
                    read_instruct_response.tStopRefresh = tThisFlipGlobal  # on global time
                    read_instruct_response.frameNStop = frameN  # exact frame index
                    # update status
                    read_instruct_response.status = FINISHED
                    read_instruct_response.stop()
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[read_instruct_response]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                diagram_condition.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in diagram_condition.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "diagram_condition" ---
        for thisComponent in diagram_condition.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for diagram_condition
        diagram_condition.tStop = globalClock.getTime(format='float')
        diagram_condition.tStopRefresh = tThisFlipGlobal
        read_instruct_response.pause()  # ensure sound has stopped at end of Routine
        # the Routine "diagram_condition" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
    # completed 3.0 repeats of 'instruct_loop'
    
    
    # --- Prepare to start Routine "instruct_review" ---
    # create an object to store info about Routine instruct_review
    instruct_review = data.Routine(
        name='instruct_review',
        components=[text_instruct_review, key_instruct_review, read_instruct_review],
    )
    instruct_review.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_instruct_review
    key_instruct_review.keys = []
    key_instruct_review.rt = []
    _key_instruct_review_allKeys = []
    read_instruct_review.setSound('resource/instruct_review.wav', hamming=True)
    read_instruct_review.setVolume(1.0, log=False)
    read_instruct_review.seek(0)
    # store start times for instruct_review
    instruct_review.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instruct_review.tStart = globalClock.getTime(format='float')
    instruct_review.status = STARTED
    instruct_review.maxDuration = None
    # keep track of which components have finished
    instruct_reviewComponents = instruct_review.components
    for thisComponent in instruct_review.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instruct_review" ---
    instruct_review.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_instruct_review* updates
        
        # if text_instruct_review is starting this frame...
        if text_instruct_review.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_instruct_review.frameNStart = frameN  # exact frame index
            text_instruct_review.tStart = t  # local t and not account for scr refresh
            text_instruct_review.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_instruct_review, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_instruct_review.status = STARTED
            text_instruct_review.setAutoDraw(True)
        
        # if text_instruct_review is active this frame...
        if text_instruct_review.status == STARTED:
            # update params
            pass
        
        # *key_instruct_review* updates
        waitOnFlip = False
        
        # if key_instruct_review is starting this frame...
        if key_instruct_review.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
            # keep track of start time/frame for later
            key_instruct_review.frameNStart = frameN  # exact frame index
            key_instruct_review.tStart = t  # local t and not account for scr refresh
            key_instruct_review.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_instruct_review, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_instruct_review.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_instruct_review.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_instruct_review.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_instruct_review.status == STARTED and not waitOnFlip:
            theseKeys = key_instruct_review.getKeys(keyList=['1'], ignoreKeys=["escape"], waitRelease=True)
            _key_instruct_review_allKeys.extend(theseKeys)
            if len(_key_instruct_review_allKeys):
                key_instruct_review.keys = _key_instruct_review_allKeys[-1].name  # just the last key pressed
                key_instruct_review.rt = _key_instruct_review_allKeys[-1].rt
                key_instruct_review.duration = _key_instruct_review_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *read_instruct_review* updates
        
        # if read_instruct_review is starting this frame...
        if read_instruct_review.status == NOT_STARTED and tThisFlip >= 0.8-frameTolerance:
            # keep track of start time/frame for later
            read_instruct_review.frameNStart = frameN  # exact frame index
            read_instruct_review.tStart = t  # local t and not account for scr refresh
            read_instruct_review.tStartRefresh = tThisFlipGlobal  # on global time
            # update status
            read_instruct_review.status = STARTED
            read_instruct_review.play(when=win)  # sync with win flip
        
        # if read_instruct_review is stopping this frame...
        if read_instruct_review.status == STARTED:
            if bool(False) or read_instruct_review.isFinished:
                # keep track of stop time/frame for later
                read_instruct_review.tStop = t  # not accounting for scr refresh
                read_instruct_review.tStopRefresh = tThisFlipGlobal  # on global time
                read_instruct_review.frameNStop = frameN  # exact frame index
                # update status
                read_instruct_review.status = FINISHED
                read_instruct_review.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[read_instruct_review]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instruct_review.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instruct_review.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruct_review" ---
    for thisComponent in instruct_review.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instruct_review
    instruct_review.tStop = globalClock.getTime(format='float')
    instruct_review.tStopRefresh = tThisFlipGlobal
    read_instruct_review.pause()  # ensure sound has stopped at end of Routine
    thisExp.nextEntry()
    # the Routine "instruct_review" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    practice_loop = data.TrialHandler2(
        name='practice_loop',
        nReps=99.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(practice_loop)  # add the loop to the experiment
    thisPractice_loop = practice_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPractice_loop.rgb)
    if thisPractice_loop != None:
        for paramName in thisPractice_loop:
            globals()[paramName] = thisPractice_loop[paramName]
    
    for thisPractice_loop in practice_loop:
        currentLoop = practice_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisPractice_loop.rgb)
        if thisPractice_loop != None:
            for paramName in thisPractice_loop:
                globals()[paramName] = thisPractice_loop[paramName]
        
        # --- Prepare to start Routine "instruct_practice_repeat" ---
        # create an object to store info about Routine instruct_practice_repeat
        instruct_practice_repeat = data.Routine(
            name='instruct_practice_repeat',
            components=[test_practice_repeat, key_practice_repeat, read_practice_repeat],
        )
        instruct_practice_repeat.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for key_practice_repeat
        key_practice_repeat.keys = []
        key_practice_repeat.rt = []
        _key_practice_repeat_allKeys = []
        read_practice_repeat.setSound('resource/instruct_practice_repeat.wav', hamming=True)
        read_practice_repeat.setVolume(1.0, log=False)
        read_practice_repeat.seek(0)
        # Run 'Begin Routine' code from skip_routine_check
        # Start a practice block
        dev.activate_line(bitmask=block_start_code)
        eyetracker.sendMessage(block_start_code)
        core.wait(0.5)  # wait 500ms before trial triggers
        
        # Skip this routine if first time doing practice
        if practice_loop.thisRepN == 0:
            continueRoutine = False
        
        # store start times for instruct_practice_repeat
        instruct_practice_repeat.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        instruct_practice_repeat.tStart = globalClock.getTime(format='float')
        instruct_practice_repeat.status = STARTED
        instruct_practice_repeat.maxDuration = None
        # keep track of which components have finished
        instruct_practice_repeatComponents = instruct_practice_repeat.components
        for thisComponent in instruct_practice_repeat.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "instruct_practice_repeat" ---
        # if trial has changed, end Routine now
        if isinstance(practice_loop, data.TrialHandler2) and thisPractice_loop.thisN != practice_loop.thisTrial.thisN:
            continueRoutine = False
        instruct_practice_repeat.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *test_practice_repeat* updates
            
            # if test_practice_repeat is starting this frame...
            if test_practice_repeat.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                test_practice_repeat.frameNStart = frameN  # exact frame index
                test_practice_repeat.tStart = t  # local t and not account for scr refresh
                test_practice_repeat.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(test_practice_repeat, 'tStartRefresh')  # time at next scr refresh
                # update status
                test_practice_repeat.status = STARTED
                test_practice_repeat.setAutoDraw(True)
            
            # if test_practice_repeat is active this frame...
            if test_practice_repeat.status == STARTED:
                # update params
                pass
            
            # *key_practice_repeat* updates
            waitOnFlip = False
            
            # if key_practice_repeat is starting this frame...
            if key_practice_repeat.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
                # keep track of start time/frame for later
                key_practice_repeat.frameNStart = frameN  # exact frame index
                key_practice_repeat.tStart = t  # local t and not account for scr refresh
                key_practice_repeat.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_practice_repeat, 'tStartRefresh')  # time at next scr refresh
                # update status
                key_practice_repeat.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_practice_repeat.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_practice_repeat.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_practice_repeat.status == STARTED and not waitOnFlip:
                theseKeys = key_practice_repeat.getKeys(keyList=['1'], ignoreKeys=["escape"], waitRelease=True)
                _key_practice_repeat_allKeys.extend(theseKeys)
                if len(_key_practice_repeat_allKeys):
                    key_practice_repeat.keys = _key_practice_repeat_allKeys[-1].name  # just the last key pressed
                    key_practice_repeat.rt = _key_practice_repeat_allKeys[-1].rt
                    key_practice_repeat.duration = _key_practice_repeat_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *read_practice_repeat* updates
            
            # if read_practice_repeat is starting this frame...
            if read_practice_repeat.status == NOT_STARTED and tThisFlip >= 0.8-frameTolerance:
                # keep track of start time/frame for later
                read_practice_repeat.frameNStart = frameN  # exact frame index
                read_practice_repeat.tStart = t  # local t and not account for scr refresh
                read_practice_repeat.tStartRefresh = tThisFlipGlobal  # on global time
                # update status
                read_practice_repeat.status = STARTED
                read_practice_repeat.play(when=win)  # sync with win flip
            
            # if read_practice_repeat is stopping this frame...
            if read_practice_repeat.status == STARTED:
                if bool(False) or read_practice_repeat.isFinished:
                    # keep track of stop time/frame for later
                    read_practice_repeat.tStop = t  # not accounting for scr refresh
                    read_practice_repeat.tStopRefresh = tThisFlipGlobal  # on global time
                    read_practice_repeat.frameNStop = frameN  # exact frame index
                    # update status
                    read_practice_repeat.status = FINISHED
                    read_practice_repeat.stop()
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[read_practice_repeat]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                instruct_practice_repeat.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in instruct_practice_repeat.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "instruct_practice_repeat" ---
        for thisComponent in instruct_practice_repeat.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for instruct_practice_repeat
        instruct_practice_repeat.tStop = globalClock.getTime(format='float')
        instruct_practice_repeat.tStopRefresh = tThisFlipGlobal
        read_practice_repeat.pause()  # ensure sound has stopped at end of Routine
        # the Routine "instruct_practice_repeat" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        practice_trials = data.TrialHandler2(
            name='practice_trials',
            nReps=n_trials_practice, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
        )
        thisExp.addLoop(practice_trials)  # add the loop to the experiment
        thisPractice_trial = practice_trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisPractice_trial.rgb)
        if thisPractice_trial != None:
            for paramName in thisPractice_trial:
                globals()[paramName] = thisPractice_trial[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisPractice_trial in practice_trials:
            currentLoop = practice_trials
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisPractice_trial.rgb)
            if thisPractice_trial != None:
                for paramName in thisPractice_trial:
                    globals()[paramName] = thisPractice_trial[paramName]
            
            # --- Prepare to start Routine "practice_setup" ---
            # create an object to store info about Routine practice_setup
            practice_setup = data.Routine(
                name='practice_setup',
                components=[],
            )
            practice_setup.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from setup_practice_trial
            # Obtain practice trial specific variables
            image_fn = image_fn_practice_list[practice_trials.thisRepN]
            image_loc = image_loc_practice_list[practice_trials.thisRepN]
            trial_type = trial_type_practice_list[practice_trials.thisRepN]
            
            # Set up prompt, filenames, location, etc.
            if trial_type == 'objsame':
                test_prompt = "Remember Object"
                image_test_fn = rng.choice(image_fn)  # one of three study objects
                image_test_loc = [0, 0]  # image location: center
                correct_resp = '1'
                
            elif trial_type == 'objdifferent':
                test_prompt = "Remember Object"
                image_test_fn = image_fn_novel_practice  # the novel object for practice
                image_test_loc = [0, 0]  # image location: center
                correct_resp = '2'
                
            elif trial_type == 'locsame':
                test_prompt = "Remember Location"
                image_test_fn = image_dot_fn  # show dot image
                image_test_loc = rng.choice(image_loc[:3])  # one of shown locations
                correct_resp = '1'
                
            elif trial_type == 'locdifferent':
                test_prompt = "Remember Location"
                image_test_fn = image_dot_fn  # show dot image
                image_test_loc = image_loc[3]  # a novel location that wasn't shown
                correct_resp = '2'
            
            elif trial_type == 'objlocsame':
                test_prompt = "Remember Object and Location"
                i = rng.choice([0, 1, 2])  # select one of the three study objects
                image_test_fn = image_fn[i]  # show the selected object
                image_test_loc = image_loc[i]  # and its corresponding location
                correct_resp = '1'
                
            elif trial_type == 'objlocdifferent':
                test_prompt = "Remember Object and Location"
                index = [0, 1, 2]
                i = rng.choice(index)  # select one of the three study objects
                _ = index.pop(i)  # remove the selected index from the list
                image_test_fn = image_fn[i]  # show the selected object
                image_test_loc = image_loc[rng.choice(index)]  # select from remaining locations
                correct_resp = '2'
            
            # store start times for practice_setup
            practice_setup.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            practice_setup.tStart = globalClock.getTime(format='float')
            practice_setup.status = STARTED
            practice_setup.maxDuration = None
            # keep track of which components have finished
            practice_setupComponents = practice_setup.components
            for thisComponent in practice_setup.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "practice_setup" ---
            # if trial has changed, end Routine now
            if isinstance(practice_trials, data.TrialHandler2) and thisPractice_trial.thisN != practice_trials.thisTrial.thisN:
                continueRoutine = False
            practice_setup.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    practice_setup.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in practice_setup.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "practice_setup" ---
            for thisComponent in practice_setup.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for practice_setup
            practice_setup.tStop = globalClock.getTime(format='float')
            practice_setup.tStopRefresh = tThisFlipGlobal
            # Run 'End Routine' code from setup_practice_trial
            thisExp.addData('image_fn_1', image_fn[0])  # adding Encoding Image Names to .csv file
            thisExp.addData('image_fn_2', image_fn[1])
            thisExp.addData('image_fn_3', image_fn[2])
            
            thisExp.addData('image_loc_1', image_loc[0])  # adding Encoding Image Locations to .csv file
            thisExp.addData('image_loc_2', image_loc[1])
            thisExp.addData('image_loc_3', image_loc[2])
            
            thisExp.addData('trial_type', trial_type)  # adding Trial Type to .csv file
            
            thisExp.addData('image_test_fn', image_test_fn)  # adding Test Image Name to .csv file
            thisExp.addData('image_test_loc', image_test_loc)  # adding Test Image Name to .csv file
            thisExp.addData('correct_response', correct_resp)  # adding Correct Response to .csv file
            
            # the Routine "practice_setup" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "trial" ---
            # create an object to store info about Routine trial
            trial = data.Routine(
                name='trial',
                components=[background, grid_outer, grid_horizontal, grid_vertical, grid_center, image_1, image_2, image_3, text_fixation, text_prompt, grid_outer_test, grid_horizontal_test, grid_vertical_test, grid_center_test, image_test, key_response_test],
            )
            trial.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            image_1.setPos([image_loc[0]])
            image_1.setImage(image_fn[0])
            image_2.setPos([image_loc[1]])
            image_2.setImage(image_fn[1])
            image_3.setPos([image_loc[2]])
            image_3.setImage(image_fn[2])
            text_prompt.setText(test_prompt)
            image_test.setPos(image_test_loc)
            image_test.setImage(image_test_fn)
            # create starting attributes for key_response_test
            key_response_test.keys = []
            key_response_test.rt = []
            _key_response_test_allKeys = []
            # Run 'Begin Routine' code from adjust_image_size
            box_size = 0.195
            scale_to_size(image_1, box_size)
            scale_to_size(image_2, box_size)
            scale_to_size(image_3, box_size)
            scale_to_size(image_test, box_size)
            
            # Run 'Begin Routine' code from trigger_trial
            background_trigger_started = False
            grid_trigger_started = False
            image_1_trigger_started = False
            image_2_trigger_started = False
            image_3_trigger_started = False
            delay_trigger_started = False
            prompt_trigger_started = False
            test_trigger_started = False
            
            # store start times for trial
            trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            trial.tStart = globalClock.getTime(format='float')
            trial.status = STARTED
            thisExp.addData('trial.started', trial.tStart)
            trial.maxDuration = None
            # keep track of which components have finished
            trialComponents = trial.components
            for thisComponent in trial.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "trial" ---
            # if trial has changed, end Routine now
            if isinstance(practice_trials, data.TrialHandler2) and thisPractice_trial.thisN != practice_trials.thisTrial.thisN:
                continueRoutine = False
            trial.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 19.15:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *background* updates
                
                # if background is starting this frame...
                if background.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    background.frameNStart = frameN  # exact frame index
                    background.tStart = t  # local t and not account for scr refresh
                    background.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(background, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'background.started')
                    # update status
                    background.status = STARTED
                    background.setAutoDraw(True)
                
                # if background is active this frame...
                if background.status == STARTED:
                    # update params
                    pass
                
                # if background is stopping this frame...
                if background.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > background.tStartRefresh + 19.15-frameTolerance:
                        # keep track of stop time/frame for later
                        background.tStop = t  # not accounting for scr refresh
                        background.tStopRefresh = tThisFlipGlobal  # on global time
                        background.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'background.stopped')
                        # update status
                        background.status = FINISHED
                        background.setAutoDraw(False)
                
                # *grid_outer* updates
                
                # if grid_outer is starting this frame...
                if grid_outer.status == NOT_STARTED and tThisFlip >= 1.0-frameTolerance:
                    # keep track of start time/frame for later
                    grid_outer.frameNStart = frameN  # exact frame index
                    grid_outer.tStart = t  # local t and not account for scr refresh
                    grid_outer.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(grid_outer, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'grid_outer.started')
                    # update status
                    grid_outer.status = STARTED
                    grid_outer.setAutoDraw(True)
                
                # if grid_outer is active this frame...
                if grid_outer.status == STARTED:
                    # update params
                    pass
                
                # if grid_outer is stopping this frame...
                if grid_outer.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > grid_outer.tStartRefresh + 4.1-frameTolerance:
                        # keep track of stop time/frame for later
                        grid_outer.tStop = t  # not accounting for scr refresh
                        grid_outer.tStopRefresh = tThisFlipGlobal  # on global time
                        grid_outer.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'grid_outer.stopped')
                        # update status
                        grid_outer.status = FINISHED
                        grid_outer.setAutoDraw(False)
                
                # *grid_horizontal* updates
                
                # if grid_horizontal is starting this frame...
                if grid_horizontal.status == NOT_STARTED and tThisFlip >= 1.0-frameTolerance:
                    # keep track of start time/frame for later
                    grid_horizontal.frameNStart = frameN  # exact frame index
                    grid_horizontal.tStart = t  # local t and not account for scr refresh
                    grid_horizontal.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(grid_horizontal, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'grid_horizontal.started')
                    # update status
                    grid_horizontal.status = STARTED
                    grid_horizontal.setAutoDraw(True)
                
                # if grid_horizontal is active this frame...
                if grid_horizontal.status == STARTED:
                    # update params
                    pass
                
                # if grid_horizontal is stopping this frame...
                if grid_horizontal.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > grid_horizontal.tStartRefresh + 4.1-frameTolerance:
                        # keep track of stop time/frame for later
                        grid_horizontal.tStop = t  # not accounting for scr refresh
                        grid_horizontal.tStopRefresh = tThisFlipGlobal  # on global time
                        grid_horizontal.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'grid_horizontal.stopped')
                        # update status
                        grid_horizontal.status = FINISHED
                        grid_horizontal.setAutoDraw(False)
                
                # *grid_vertical* updates
                
                # if grid_vertical is starting this frame...
                if grid_vertical.status == NOT_STARTED and tThisFlip >= 1.0-frameTolerance:
                    # keep track of start time/frame for later
                    grid_vertical.frameNStart = frameN  # exact frame index
                    grid_vertical.tStart = t  # local t and not account for scr refresh
                    grid_vertical.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(grid_vertical, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'grid_vertical.started')
                    # update status
                    grid_vertical.status = STARTED
                    grid_vertical.setAutoDraw(True)
                
                # if grid_vertical is active this frame...
                if grid_vertical.status == STARTED:
                    # update params
                    pass
                
                # if grid_vertical is stopping this frame...
                if grid_vertical.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > grid_vertical.tStartRefresh + 4.1-frameTolerance:
                        # keep track of stop time/frame for later
                        grid_vertical.tStop = t  # not accounting for scr refresh
                        grid_vertical.tStopRefresh = tThisFlipGlobal  # on global time
                        grid_vertical.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'grid_vertical.stopped')
                        # update status
                        grid_vertical.status = FINISHED
                        grid_vertical.setAutoDraw(False)
                
                # *grid_center* updates
                
                # if grid_center is starting this frame...
                if grid_center.status == NOT_STARTED and tThisFlip >= 1.0-frameTolerance:
                    # keep track of start time/frame for later
                    grid_center.frameNStart = frameN  # exact frame index
                    grid_center.tStart = t  # local t and not account for scr refresh
                    grid_center.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(grid_center, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'grid_center.started')
                    # update status
                    grid_center.status = STARTED
                    grid_center.setAutoDraw(True)
                
                # if grid_center is active this frame...
                if grid_center.status == STARTED:
                    # update params
                    pass
                
                # if grid_center is stopping this frame...
                if grid_center.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > grid_center.tStartRefresh + 4.1-frameTolerance:
                        # keep track of stop time/frame for later
                        grid_center.tStop = t  # not accounting for scr refresh
                        grid_center.tStopRefresh = tThisFlipGlobal  # on global time
                        grid_center.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'grid_center.stopped')
                        # update status
                        grid_center.status = FINISHED
                        grid_center.setAutoDraw(False)
                
                # *image_1* updates
                
                # if image_1 is starting this frame...
                if image_1.status == NOT_STARTED and tThisFlip >= 2.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_1.frameNStart = frameN  # exact frame index
                    image_1.tStart = t  # local t and not account for scr refresh
                    image_1.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_1, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_1.started')
                    # update status
                    image_1.status = STARTED
                    image_1.setAutoDraw(True)
                
                # if image_1 is active this frame...
                if image_1.status == STARTED:
                    # update params
                    pass
                
                # if image_1 is stopping this frame...
                if image_1.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_1.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        image_1.tStop = t  # not accounting for scr refresh
                        image_1.tStopRefresh = tThisFlipGlobal  # on global time
                        image_1.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_1.stopped')
                        # update status
                        image_1.status = FINISHED
                        image_1.setAutoDraw(False)
                
                # *image_2* updates
                
                # if image_2 is starting this frame...
                if image_2.status == NOT_STARTED and tThisFlip >= 3.05-frameTolerance:
                    # keep track of start time/frame for later
                    image_2.frameNStart = frameN  # exact frame index
                    image_2.tStart = t  # local t and not account for scr refresh
                    image_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_2.started')
                    # update status
                    image_2.status = STARTED
                    image_2.setAutoDraw(True)
                
                # if image_2 is active this frame...
                if image_2.status == STARTED:
                    # update params
                    pass
                
                # if image_2 is stopping this frame...
                if image_2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_2.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        image_2.tStop = t  # not accounting for scr refresh
                        image_2.tStopRefresh = tThisFlipGlobal  # on global time
                        image_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_2.stopped')
                        # update status
                        image_2.status = FINISHED
                        image_2.setAutoDraw(False)
                
                # *image_3* updates
                
                # if image_3 is starting this frame...
                if image_3.status == NOT_STARTED and tThisFlip >= 4.1-frameTolerance:
                    # keep track of start time/frame for later
                    image_3.frameNStart = frameN  # exact frame index
                    image_3.tStart = t  # local t and not account for scr refresh
                    image_3.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_3, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_3.started')
                    # update status
                    image_3.status = STARTED
                    image_3.setAutoDraw(True)
                
                # if image_3 is active this frame...
                if image_3.status == STARTED:
                    # update params
                    pass
                
                # if image_3 is stopping this frame...
                if image_3.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_3.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        image_3.tStop = t  # not accounting for scr refresh
                        image_3.tStopRefresh = tThisFlipGlobal  # on global time
                        image_3.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_3.stopped')
                        # update status
                        image_3.status = FINISHED
                        image_3.setAutoDraw(False)
                
                # *text_fixation* updates
                
                # if text_fixation is starting this frame...
                if text_fixation.status == NOT_STARTED and tThisFlip >= 5.15-frameTolerance:
                    # keep track of start time/frame for later
                    text_fixation.frameNStart = frameN  # exact frame index
                    text_fixation.tStart = t  # local t and not account for scr refresh
                    text_fixation.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_fixation, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_fixation.started')
                    # update status
                    text_fixation.status = STARTED
                    text_fixation.setAutoDraw(True)
                
                # if text_fixation is active this frame...
                if text_fixation.status == STARTED:
                    # update params
                    pass
                
                # if text_fixation is stopping this frame...
                if text_fixation.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_fixation.tStartRefresh + 8.0-frameTolerance:
                        # keep track of stop time/frame for later
                        text_fixation.tStop = t  # not accounting for scr refresh
                        text_fixation.tStopRefresh = tThisFlipGlobal  # on global time
                        text_fixation.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_fixation.stopped')
                        # update status
                        text_fixation.status = FINISHED
                        text_fixation.setAutoDraw(False)
                
                # *text_prompt* updates
                
                # if text_prompt is starting this frame...
                if text_prompt.status == NOT_STARTED and tThisFlip >= 13.15-frameTolerance:
                    # keep track of start time/frame for later
                    text_prompt.frameNStart = frameN  # exact frame index
                    text_prompt.tStart = t  # local t and not account for scr refresh
                    text_prompt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_prompt, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_prompt.started')
                    # update status
                    text_prompt.status = STARTED
                    text_prompt.setAutoDraw(True)
                
                # if text_prompt is active this frame...
                if text_prompt.status == STARTED:
                    # update params
                    pass
                
                # if text_prompt is stopping this frame...
                if text_prompt.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_prompt.tStartRefresh + 2.0-frameTolerance:
                        # keep track of stop time/frame for later
                        text_prompt.tStop = t  # not accounting for scr refresh
                        text_prompt.tStopRefresh = tThisFlipGlobal  # on global time
                        text_prompt.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_prompt.stopped')
                        # update status
                        text_prompt.status = FINISHED
                        text_prompt.setAutoDraw(False)
                
                # *grid_outer_test* updates
                
                # if grid_outer_test is starting this frame...
                if grid_outer_test.status == NOT_STARTED and tThisFlip >= 15.15-frameTolerance:
                    # keep track of start time/frame for later
                    grid_outer_test.frameNStart = frameN  # exact frame index
                    grid_outer_test.tStart = t  # local t and not account for scr refresh
                    grid_outer_test.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(grid_outer_test, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'grid_outer_test.started')
                    # update status
                    grid_outer_test.status = STARTED
                    grid_outer_test.setAutoDraw(True)
                
                # if grid_outer_test is active this frame...
                if grid_outer_test.status == STARTED:
                    # update params
                    pass
                
                # if grid_outer_test is stopping this frame...
                if grid_outer_test.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > grid_outer_test.tStartRefresh + 4.0-frameTolerance:
                        # keep track of stop time/frame for later
                        grid_outer_test.tStop = t  # not accounting for scr refresh
                        grid_outer_test.tStopRefresh = tThisFlipGlobal  # on global time
                        grid_outer_test.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'grid_outer_test.stopped')
                        # update status
                        grid_outer_test.status = FINISHED
                        grid_outer_test.setAutoDraw(False)
                
                # *grid_horizontal_test* updates
                
                # if grid_horizontal_test is starting this frame...
                if grid_horizontal_test.status == NOT_STARTED and tThisFlip >= 15.15-frameTolerance:
                    # keep track of start time/frame for later
                    grid_horizontal_test.frameNStart = frameN  # exact frame index
                    grid_horizontal_test.tStart = t  # local t and not account for scr refresh
                    grid_horizontal_test.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(grid_horizontal_test, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'grid_horizontal_test.started')
                    # update status
                    grid_horizontal_test.status = STARTED
                    grid_horizontal_test.setAutoDraw(True)
                
                # if grid_horizontal_test is active this frame...
                if grid_horizontal_test.status == STARTED:
                    # update params
                    pass
                
                # if grid_horizontal_test is stopping this frame...
                if grid_horizontal_test.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > grid_horizontal_test.tStartRefresh + 4.0-frameTolerance:
                        # keep track of stop time/frame for later
                        grid_horizontal_test.tStop = t  # not accounting for scr refresh
                        grid_horizontal_test.tStopRefresh = tThisFlipGlobal  # on global time
                        grid_horizontal_test.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'grid_horizontal_test.stopped')
                        # update status
                        grid_horizontal_test.status = FINISHED
                        grid_horizontal_test.setAutoDraw(False)
                
                # *grid_vertical_test* updates
                
                # if grid_vertical_test is starting this frame...
                if grid_vertical_test.status == NOT_STARTED and tThisFlip >= 15.15-frameTolerance:
                    # keep track of start time/frame for later
                    grid_vertical_test.frameNStart = frameN  # exact frame index
                    grid_vertical_test.tStart = t  # local t and not account for scr refresh
                    grid_vertical_test.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(grid_vertical_test, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'grid_vertical_test.started')
                    # update status
                    grid_vertical_test.status = STARTED
                    grid_vertical_test.setAutoDraw(True)
                
                # if grid_vertical_test is active this frame...
                if grid_vertical_test.status == STARTED:
                    # update params
                    pass
                
                # if grid_vertical_test is stopping this frame...
                if grid_vertical_test.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > grid_vertical_test.tStartRefresh + 4.0-frameTolerance:
                        # keep track of stop time/frame for later
                        grid_vertical_test.tStop = t  # not accounting for scr refresh
                        grid_vertical_test.tStopRefresh = tThisFlipGlobal  # on global time
                        grid_vertical_test.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'grid_vertical_test.stopped')
                        # update status
                        grid_vertical_test.status = FINISHED
                        grid_vertical_test.setAutoDraw(False)
                
                # *grid_center_test* updates
                
                # if grid_center_test is starting this frame...
                if grid_center_test.status == NOT_STARTED and tThisFlip >= 15.15-frameTolerance:
                    # keep track of start time/frame for later
                    grid_center_test.frameNStart = frameN  # exact frame index
                    grid_center_test.tStart = t  # local t and not account for scr refresh
                    grid_center_test.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(grid_center_test, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'grid_center_test.started')
                    # update status
                    grid_center_test.status = STARTED
                    grid_center_test.setAutoDraw(True)
                
                # if grid_center_test is active this frame...
                if grid_center_test.status == STARTED:
                    # update params
                    pass
                
                # if grid_center_test is stopping this frame...
                if grid_center_test.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > grid_center_test.tStartRefresh + 4.0-frameTolerance:
                        # keep track of stop time/frame for later
                        grid_center_test.tStop = t  # not accounting for scr refresh
                        grid_center_test.tStopRefresh = tThisFlipGlobal  # on global time
                        grid_center_test.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'grid_center_test.stopped')
                        # update status
                        grid_center_test.status = FINISHED
                        grid_center_test.setAutoDraw(False)
                
                # *image_test* updates
                
                # if image_test is starting this frame...
                if image_test.status == NOT_STARTED and tThisFlip >= 15.15-frameTolerance:
                    # keep track of start time/frame for later
                    image_test.frameNStart = frameN  # exact frame index
                    image_test.tStart = t  # local t and not account for scr refresh
                    image_test.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_test, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_test.started')
                    # update status
                    image_test.status = STARTED
                    image_test.setAutoDraw(True)
                
                # if image_test is active this frame...
                if image_test.status == STARTED:
                    # update params
                    pass
                
                # if image_test is stopping this frame...
                if image_test.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_test.tStartRefresh + 4.0-frameTolerance:
                        # keep track of stop time/frame for later
                        image_test.tStop = t  # not accounting for scr refresh
                        image_test.tStopRefresh = tThisFlipGlobal  # on global time
                        image_test.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_test.stopped')
                        # update status
                        image_test.status = FINISHED
                        image_test.setAutoDraw(False)
                
                # *key_response_test* updates
                waitOnFlip = False
                
                # if key_response_test is starting this frame...
                if key_response_test.status == NOT_STARTED and tThisFlip >= 15.15-frameTolerance:
                    # keep track of start time/frame for later
                    key_response_test.frameNStart = frameN  # exact frame index
                    key_response_test.tStart = t  # local t and not account for scr refresh
                    key_response_test.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_response_test, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_response_test.started')
                    # update status
                    key_response_test.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_response_test.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_response_test.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if key_response_test is stopping this frame...
                if key_response_test.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > key_response_test.tStartRefresh + 4.0-frameTolerance:
                        # keep track of stop time/frame for later
                        key_response_test.tStop = t  # not accounting for scr refresh
                        key_response_test.tStopRefresh = tThisFlipGlobal  # on global time
                        key_response_test.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'key_response_test.stopped')
                        # update status
                        key_response_test.status = FINISHED
                        key_response_test.status = FINISHED
                if key_response_test.status == STARTED and not waitOnFlip:
                    theseKeys = key_response_test.getKeys(keyList=['1', '2'], ignoreKeys=["escape"], waitRelease=False)
                    _key_response_test_allKeys.extend(theseKeys)
                    if len(_key_response_test_allKeys):
                        key_response_test.keys = _key_response_test_allKeys[-1].name  # just the last key pressed
                        key_response_test.rt = _key_response_test_allKeys[-1].rt
                        key_response_test.duration = _key_response_test_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                # Run 'Each Frame' code from trigger_trial
                if background.status == STARTED and not background_trigger_started:
                    win.callOnFlip(dev.activate_line, bitmask=background_start_code)
                    win.callOnFlip(eyetracker.sendMessage, background_start_code)
                    background_trigger_started = True
                
                if grid_outer.status == STARTED and not grid_trigger_started:
                    win.callOnFlip(dev.activate_line, bitmask=grid_start_code)
                    win.callOnFlip(eyetracker.sendMessage, grid_start_code)
                    grid_trigger_started = True
                
                if image_1.status == STARTED and not image_1_trigger_started:
                    win.callOnFlip(dev.activate_line, bitmask=image_1_start_code)
                    win.callOnFlip(eyetracker.sendMessage, image_1_start_code)
                    image_1_trigger_started = True
                
                if image_2.status == STARTED and not image_2_trigger_started:
                    win.callOnFlip(dev.activate_line, bitmask=image_2_start_code)
                    win.callOnFlip(eyetracker.sendMessage, image_2_start_code)
                    image_2_trigger_started = True
                
                if image_3.status == STARTED and not image_3_trigger_started:
                    win.callOnFlip(dev.activate_line, bitmask=image_3_start_code)
                    win.callOnFlip(eyetracker.sendMessage, image_3_start_code)
                    image_3_trigger_started = True
                
                if text_fixation.status == STARTED and not delay_trigger_started:
                    win.callOnFlip(dev.activate_line, bitmask=delay_start_code)
                    win.callOnFlip(eyetracker.sendMessage, delay_start_code)
                    delay_trigger_started = True
                
                if text_prompt.status == STARTED and not prompt_trigger_started:
                    win.callOnFlip(dev.activate_line, bitmask=prompt_start_code)
                    win.callOnFlip(eyetracker.sendMessage, prompt_start_code)
                    prompt_trigger_started = True
                
                if image_test.status == STARTED and not test_trigger_started:
                    win.callOnFlip(dev.activate_line, bitmask=test_start_code)
                    win.callOnFlip(eyetracker.sendMessage, test_start_code)
                    test_trigger_started = True
                
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    trial.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trial.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial" ---
            for thisComponent in trial.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for trial
            trial.tStop = globalClock.getTime(format='float')
            trial.tStopRefresh = tThisFlipGlobal
            thisExp.addData('trial.stopped', trial.tStop)
            # check responses
            if key_response_test.keys in ['', [], None]:  # No response was made
                key_response_test.keys = None
            practice_trials.addData('key_response_test.keys',key_response_test.keys)
            if key_response_test.keys != None:  # we had a response
                practice_trials.addData('key_response_test.rt', key_response_test.rt)
                practice_trials.addData('key_response_test.duration', key_response_test.duration)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if trial.maxDurationReached:
                routineTimer.addTime(-trial.maxDuration)
            elif trial.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-19.150000)
            
            # --- Prepare to start Routine "practice_feedback" ---
            # create an object to store info about Routine practice_feedback
            practice_feedback = data.Routine(
                name='practice_feedback',
                components=[background_feedback, text_feedback],
            )
            practice_feedback.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from set_feedback_text
            if not key_response_test.keys:  # no key pressed
                feedback_text = 'Respond Faster'
                feedback_text_color = [-1, -1, -1]  # black
            elif key_response_test.keys == correct_resp:
                feedback_text = 'Correct'
                feedback_text_color = [-1, 1, -1]  # green
            else:
                feedback_text = 'Incorrect'
                feedback_text_color = [1, -1, -1]  # red
            
            text_feedback.setColor(feedback_text_color, colorSpace='rgb')
            text_feedback.setText(feedback_text
            )
            # store start times for practice_feedback
            practice_feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            practice_feedback.tStart = globalClock.getTime(format='float')
            practice_feedback.status = STARTED
            practice_feedback.maxDuration = None
            # keep track of which components have finished
            practice_feedbackComponents = practice_feedback.components
            for thisComponent in practice_feedback.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "practice_feedback" ---
            # if trial has changed, end Routine now
            if isinstance(practice_trials, data.TrialHandler2) and thisPractice_trial.thisN != practice_trials.thisTrial.thisN:
                continueRoutine = False
            practice_feedback.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 2.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *background_feedback* updates
                
                # if background_feedback is starting this frame...
                if background_feedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    background_feedback.frameNStart = frameN  # exact frame index
                    background_feedback.tStart = t  # local t and not account for scr refresh
                    background_feedback.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(background_feedback, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    background_feedback.status = STARTED
                    background_feedback.setAutoDraw(True)
                
                # if background_feedback is active this frame...
                if background_feedback.status == STARTED:
                    # update params
                    pass
                
                # if background_feedback is stopping this frame...
                if background_feedback.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > background_feedback.tStartRefresh + 2-frameTolerance:
                        # keep track of stop time/frame for later
                        background_feedback.tStop = t  # not accounting for scr refresh
                        background_feedback.tStopRefresh = tThisFlipGlobal  # on global time
                        background_feedback.frameNStop = frameN  # exact frame index
                        # update status
                        background_feedback.status = FINISHED
                        background_feedback.setAutoDraw(False)
                
                # *text_feedback* updates
                
                # if text_feedback is starting this frame...
                if text_feedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_feedback.frameNStart = frameN  # exact frame index
                    text_feedback.tStart = t  # local t and not account for scr refresh
                    text_feedback.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_feedback, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    text_feedback.status = STARTED
                    text_feedback.setAutoDraw(True)
                
                # if text_feedback is active this frame...
                if text_feedback.status == STARTED:
                    # update params
                    pass
                
                # if text_feedback is stopping this frame...
                if text_feedback.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_feedback.tStartRefresh + 2.0-frameTolerance:
                        # keep track of stop time/frame for later
                        text_feedback.tStop = t  # not accounting for scr refresh
                        text_feedback.tStopRefresh = tThisFlipGlobal  # on global time
                        text_feedback.frameNStop = frameN  # exact frame index
                        # update status
                        text_feedback.status = FINISHED
                        text_feedback.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    practice_feedback.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in practice_feedback.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "practice_feedback" ---
            for thisComponent in practice_feedback.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for practice_feedback
            practice_feedback.tStop = globalClock.getTime(format='float')
            practice_feedback.tStopRefresh = tThisFlipGlobal
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if practice_feedback.maxDurationReached:
                routineTimer.addTime(-practice_feedback.maxDuration)
            elif practice_feedback.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-2.000000)
            thisExp.nextEntry()
            
        # completed n_trials_practice repeats of 'practice_trials'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # get names of stimulus parameters
        if practice_trials.trialList in ([], [None], None):
            params = []
        else:
            params = practice_trials.trialList[0].keys()
        # save data for this loop
        practice_trials.saveAsText(filename + 'practice_trials.csv', delim=',',
            stimOut=params,
            dataOut=['n','all_mean','all_std', 'all_raw'])
        
        # --- Prepare to start Routine "practice_checkpoint" ---
        # create an object to store info about Routine practice_checkpoint
        practice_checkpoint = data.Routine(
            name='practice_checkpoint',
            components=[text_checkpoint, key_checkpoint, read_checkpoint],
        )
        practice_checkpoint.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for key_checkpoint
        key_checkpoint.keys = []
        key_checkpoint.rt = []
        _key_checkpoint_allKeys = []
        read_checkpoint.setSound('resource/instruct_checkpoint.wav', hamming=True)
        read_checkpoint.setVolume(1.0, log=False)
        read_checkpoint.seek(0)
        # Run 'Begin Routine' code from code_checkpoint
        # End of practice block
        dev.activate_line(bitmask=block_end_code)
        eyetracker.sendMessage(block_end_code)
        # no need to wait 500ms as this routine waits for experimenter key press
        
        # store start times for practice_checkpoint
        practice_checkpoint.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        practice_checkpoint.tStart = globalClock.getTime(format='float')
        practice_checkpoint.status = STARTED
        practice_checkpoint.maxDuration = None
        # keep track of which components have finished
        practice_checkpointComponents = practice_checkpoint.components
        for thisComponent in practice_checkpoint.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "practice_checkpoint" ---
        # if trial has changed, end Routine now
        if isinstance(practice_loop, data.TrialHandler2) and thisPractice_loop.thisN != practice_loop.thisTrial.thisN:
            continueRoutine = False
        practice_checkpoint.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_checkpoint* updates
            
            # if text_checkpoint is starting this frame...
            if text_checkpoint.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_checkpoint.frameNStart = frameN  # exact frame index
                text_checkpoint.tStart = t  # local t and not account for scr refresh
                text_checkpoint.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_checkpoint, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_checkpoint.status = STARTED
                text_checkpoint.setAutoDraw(True)
            
            # if text_checkpoint is active this frame...
            if text_checkpoint.status == STARTED:
                # update params
                pass
            
            # *key_checkpoint* updates
            waitOnFlip = False
            
            # if key_checkpoint is starting this frame...
            if key_checkpoint.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
                # keep track of start time/frame for later
                key_checkpoint.frameNStart = frameN  # exact frame index
                key_checkpoint.tStart = t  # local t and not account for scr refresh
                key_checkpoint.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_checkpoint, 'tStartRefresh')  # time at next scr refresh
                # update status
                key_checkpoint.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_checkpoint.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_checkpoint.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_checkpoint.status == STARTED and not waitOnFlip:
                theseKeys = key_checkpoint.getKeys(keyList=['r', 'o'], ignoreKeys=["escape"], waitRelease=True)
                _key_checkpoint_allKeys.extend(theseKeys)
                if len(_key_checkpoint_allKeys):
                    key_checkpoint.keys = _key_checkpoint_allKeys[-1].name  # just the last key pressed
                    key_checkpoint.rt = _key_checkpoint_allKeys[-1].rt
                    key_checkpoint.duration = _key_checkpoint_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *read_checkpoint* updates
            
            # if read_checkpoint is starting this frame...
            if read_checkpoint.status == NOT_STARTED and tThisFlip >= 0.4-frameTolerance:
                # keep track of start time/frame for later
                read_checkpoint.frameNStart = frameN  # exact frame index
                read_checkpoint.tStart = t  # local t and not account for scr refresh
                read_checkpoint.tStartRefresh = tThisFlipGlobal  # on global time
                # update status
                read_checkpoint.status = STARTED
                read_checkpoint.play(when=win)  # sync with win flip
            
            # if read_checkpoint is stopping this frame...
            if read_checkpoint.status == STARTED:
                if bool(False) or read_checkpoint.isFinished:
                    # keep track of stop time/frame for later
                    read_checkpoint.tStop = t  # not accounting for scr refresh
                    read_checkpoint.tStopRefresh = tThisFlipGlobal  # on global time
                    read_checkpoint.frameNStop = frameN  # exact frame index
                    # update status
                    read_checkpoint.status = FINISHED
                    read_checkpoint.stop()
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[read_checkpoint]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                practice_checkpoint.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in practice_checkpoint.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "practice_checkpoint" ---
        for thisComponent in practice_checkpoint.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for practice_checkpoint
        practice_checkpoint.tStop = globalClock.getTime(format='float')
        practice_checkpoint.tStopRefresh = tThisFlipGlobal
        read_checkpoint.pause()  # ensure sound has stopped at end of Routine
        # Run 'End Routine' code from code_checkpoint
        if key_checkpoint.keys == 'o':  # proceed to main experiment
            practice_loop.finished = True
        
        # the Routine "practice_checkpoint" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
    # completed 99.0 repeats of 'practice_loop'
    
    
    # --- Prepare to start Routine "instruct_begin" ---
    # create an object to store info about Routine instruct_begin
    instruct_begin = data.Routine(
        name='instruct_begin',
        components=[text_instruct_begin, key_instruct_begin, read_instruct_begin],
    )
    instruct_begin.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_instruct_begin
    key_instruct_begin.keys = []
    key_instruct_begin.rt = []
    _key_instruct_begin_allKeys = []
    read_instruct_begin.setSound('resource/instruct_begin.wav', hamming=True)
    read_instruct_begin.setVolume(1.0, log=False)
    read_instruct_begin.seek(0)
    # Run 'Begin Routine' code from trigger_trial_block
    # Beginning of main experiment trial block
    dev.activate_line(bitmask=block_start_code)
    eyetracker.sendMessage(block_start_code)
    # no need to wait 500ms as this routine waits for subject key press
    
    # store start times for instruct_begin
    instruct_begin.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instruct_begin.tStart = globalClock.getTime(format='float')
    instruct_begin.status = STARTED
    instruct_begin.maxDuration = None
    # keep track of which components have finished
    instruct_beginComponents = instruct_begin.components
    for thisComponent in instruct_begin.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instruct_begin" ---
    instruct_begin.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_instruct_begin* updates
        
        # if text_instruct_begin is starting this frame...
        if text_instruct_begin.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_instruct_begin.frameNStart = frameN  # exact frame index
            text_instruct_begin.tStart = t  # local t and not account for scr refresh
            text_instruct_begin.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_instruct_begin, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_instruct_begin.status = STARTED
            text_instruct_begin.setAutoDraw(True)
        
        # if text_instruct_begin is active this frame...
        if text_instruct_begin.status == STARTED:
            # update params
            pass
        
        # *key_instruct_begin* updates
        waitOnFlip = False
        
        # if key_instruct_begin is starting this frame...
        if key_instruct_begin.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
            # keep track of start time/frame for later
            key_instruct_begin.frameNStart = frameN  # exact frame index
            key_instruct_begin.tStart = t  # local t and not account for scr refresh
            key_instruct_begin.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_instruct_begin, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_instruct_begin.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_instruct_begin.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_instruct_begin.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_instruct_begin.status == STARTED and not waitOnFlip:
            theseKeys = key_instruct_begin.getKeys(keyList=['1'], ignoreKeys=["escape"], waitRelease=True)
            _key_instruct_begin_allKeys.extend(theseKeys)
            if len(_key_instruct_begin_allKeys):
                key_instruct_begin.keys = _key_instruct_begin_allKeys[-1].name  # just the last key pressed
                key_instruct_begin.rt = _key_instruct_begin_allKeys[-1].rt
                key_instruct_begin.duration = _key_instruct_begin_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *read_instruct_begin* updates
        
        # if read_instruct_begin is starting this frame...
        if read_instruct_begin.status == NOT_STARTED and tThisFlip >= 0.8-frameTolerance:
            # keep track of start time/frame for later
            read_instruct_begin.frameNStart = frameN  # exact frame index
            read_instruct_begin.tStart = t  # local t and not account for scr refresh
            read_instruct_begin.tStartRefresh = tThisFlipGlobal  # on global time
            # update status
            read_instruct_begin.status = STARTED
            read_instruct_begin.play(when=win)  # sync with win flip
        
        # if read_instruct_begin is stopping this frame...
        if read_instruct_begin.status == STARTED:
            if bool(False) or read_instruct_begin.isFinished:
                # keep track of stop time/frame for later
                read_instruct_begin.tStop = t  # not accounting for scr refresh
                read_instruct_begin.tStopRefresh = tThisFlipGlobal  # on global time
                read_instruct_begin.frameNStop = frameN  # exact frame index
                # update status
                read_instruct_begin.status = FINISHED
                read_instruct_begin.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[read_instruct_begin]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instruct_begin.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instruct_begin.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruct_begin" ---
    for thisComponent in instruct_begin.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instruct_begin
    instruct_begin.tStop = globalClock.getTime(format='float')
    instruct_begin.tStopRefresh = tThisFlipGlobal
    read_instruct_begin.pause()  # ensure sound has stopped at end of Routine
    thisExp.nextEntry()
    # the Routine "instruct_begin" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=n_trials, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "trial_setup" ---
        # create an object to store info about Routine trial_setup
        trial_setup = data.Routine(
            name='trial_setup',
            components=[],
        )
        trial_setup.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from setup_image_trial
        # Obtain trial specific variables
        image_fn = image_fn_list[trials.thisRepN]
        image_loc = image_loc_list[trials.thisRepN]
        trial_type = trial_type_list[trials.thisRepN]
        
        # Set up prompt, filenames, location, etc.
        if trial_type == 'objsame':
            test_prompt = "Remember Object"
            image_test_fn = rng.choice(image_fn)  # one of three study objects
            image_test_loc = [0, 0]  # image location: center
            correct_resp = '1'
            
        elif trial_type == 'objdifferent':
            test_prompt = "Remember Object"
            image_test_fn = image_fn_novel_list.pop()  # get a new novel object
            image_test_loc = [0, 0]  # image location: center
            correct_resp = '2'
            
        elif trial_type == 'locsame':
            test_prompt = "Remember Location"
            image_test_fn = image_dot_fn  # show dot image
            image_test_loc = rng.choice(image_loc[:3])  # one of shown locations
            correct_resp = '1'
            
        elif trial_type == 'locdifferent':
            test_prompt = "Remember Location"
            image_test_fn = image_dot_fn  # show dot image
            image_test_loc = image_loc[3]  # a novel location that wasn't shown
            correct_resp = '2'
        
        elif trial_type == 'objlocsame':
            test_prompt = "Remember Object and Location"
            i = rng.choice([0, 1, 2])  # select one of the three study objects
            image_test_fn = image_fn[i]  # show the selected object
            image_test_loc = image_loc[i]  # and its corresponding location
            correct_resp = '1'
            
        elif trial_type == 'objlocdifferent':
            test_prompt = "Remember Object and Location"
            index = [0, 1, 2]
            i = rng.choice(index)  # select one of the three study objects
            _ = index.pop(i)  # remove the selected index from the list
            image_test_fn = image_fn[i]  # show the selected object
            image_test_loc = image_loc[rng.choice(index)]  # select from remaining locations
            correct_resp = '2'
        
        # store start times for trial_setup
        trial_setup.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trial_setup.tStart = globalClock.getTime(format='float')
        trial_setup.status = STARTED
        trial_setup.maxDuration = None
        # keep track of which components have finished
        trial_setupComponents = trial_setup.components
        for thisComponent in trial_setup.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial_setup" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        trial_setup.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                trial_setup.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trial_setup.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial_setup" ---
        for thisComponent in trial_setup.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trial_setup
        trial_setup.tStop = globalClock.getTime(format='float')
        trial_setup.tStopRefresh = tThisFlipGlobal
        # Run 'End Routine' code from setup_image_trial
        thisExp.addData('image_fn_1', image_fn[0])  # adding Encoding Image Names to .csv file
        thisExp.addData('image_fn_2', image_fn[1])
        thisExp.addData('image_fn_3', image_fn[2])
        
        thisExp.addData('image_loc_1', image_loc[0])  # adding Encoding Image Locations to .csv file
        thisExp.addData('image_loc_2', image_loc[1])
        thisExp.addData('image_loc_3', image_loc[2])
        
        thisExp.addData('trial_type', trial_type)  # adding Trial Type to .csv file
        
        thisExp.addData('image_test_fn', image_test_fn)  # adding Test Image Name to .csv file
        thisExp.addData('image_test_loc', image_test_loc)  # adding Test Image Name to .csv file
        thisExp.addData('correct_response', correct_resp)  # adding Correct Response to .csv file
        
        # the Routine "trial_setup" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "trial" ---
        # create an object to store info about Routine trial
        trial = data.Routine(
            name='trial',
            components=[background, grid_outer, grid_horizontal, grid_vertical, grid_center, image_1, image_2, image_3, text_fixation, text_prompt, grid_outer_test, grid_horizontal_test, grid_vertical_test, grid_center_test, image_test, key_response_test],
        )
        trial.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        image_1.setPos([image_loc[0]])
        image_1.setImage(image_fn[0])
        image_2.setPos([image_loc[1]])
        image_2.setImage(image_fn[1])
        image_3.setPos([image_loc[2]])
        image_3.setImage(image_fn[2])
        text_prompt.setText(test_prompt)
        image_test.setPos(image_test_loc)
        image_test.setImage(image_test_fn)
        # create starting attributes for key_response_test
        key_response_test.keys = []
        key_response_test.rt = []
        _key_response_test_allKeys = []
        # Run 'Begin Routine' code from adjust_image_size
        box_size = 0.195
        scale_to_size(image_1, box_size)
        scale_to_size(image_2, box_size)
        scale_to_size(image_3, box_size)
        scale_to_size(image_test, box_size)
        
        # Run 'Begin Routine' code from trigger_trial
        background_trigger_started = False
        grid_trigger_started = False
        image_1_trigger_started = False
        image_2_trigger_started = False
        image_3_trigger_started = False
        delay_trigger_started = False
        prompt_trigger_started = False
        test_trigger_started = False
        
        # store start times for trial
        trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trial.tStart = globalClock.getTime(format='float')
        trial.status = STARTED
        thisExp.addData('trial.started', trial.tStart)
        trial.maxDuration = None
        # keep track of which components have finished
        trialComponents = trial.components
        for thisComponent in trial.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        trial.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 19.15:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *background* updates
            
            # if background is starting this frame...
            if background.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                background.frameNStart = frameN  # exact frame index
                background.tStart = t  # local t and not account for scr refresh
                background.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(background, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'background.started')
                # update status
                background.status = STARTED
                background.setAutoDraw(True)
            
            # if background is active this frame...
            if background.status == STARTED:
                # update params
                pass
            
            # if background is stopping this frame...
            if background.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > background.tStartRefresh + 19.15-frameTolerance:
                    # keep track of stop time/frame for later
                    background.tStop = t  # not accounting for scr refresh
                    background.tStopRefresh = tThisFlipGlobal  # on global time
                    background.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'background.stopped')
                    # update status
                    background.status = FINISHED
                    background.setAutoDraw(False)
            
            # *grid_outer* updates
            
            # if grid_outer is starting this frame...
            if grid_outer.status == NOT_STARTED and tThisFlip >= 1.0-frameTolerance:
                # keep track of start time/frame for later
                grid_outer.frameNStart = frameN  # exact frame index
                grid_outer.tStart = t  # local t and not account for scr refresh
                grid_outer.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(grid_outer, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'grid_outer.started')
                # update status
                grid_outer.status = STARTED
                grid_outer.setAutoDraw(True)
            
            # if grid_outer is active this frame...
            if grid_outer.status == STARTED:
                # update params
                pass
            
            # if grid_outer is stopping this frame...
            if grid_outer.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > grid_outer.tStartRefresh + 4.1-frameTolerance:
                    # keep track of stop time/frame for later
                    grid_outer.tStop = t  # not accounting for scr refresh
                    grid_outer.tStopRefresh = tThisFlipGlobal  # on global time
                    grid_outer.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'grid_outer.stopped')
                    # update status
                    grid_outer.status = FINISHED
                    grid_outer.setAutoDraw(False)
            
            # *grid_horizontal* updates
            
            # if grid_horizontal is starting this frame...
            if grid_horizontal.status == NOT_STARTED and tThisFlip >= 1.0-frameTolerance:
                # keep track of start time/frame for later
                grid_horizontal.frameNStart = frameN  # exact frame index
                grid_horizontal.tStart = t  # local t and not account for scr refresh
                grid_horizontal.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(grid_horizontal, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'grid_horizontal.started')
                # update status
                grid_horizontal.status = STARTED
                grid_horizontal.setAutoDraw(True)
            
            # if grid_horizontal is active this frame...
            if grid_horizontal.status == STARTED:
                # update params
                pass
            
            # if grid_horizontal is stopping this frame...
            if grid_horizontal.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > grid_horizontal.tStartRefresh + 4.1-frameTolerance:
                    # keep track of stop time/frame for later
                    grid_horizontal.tStop = t  # not accounting for scr refresh
                    grid_horizontal.tStopRefresh = tThisFlipGlobal  # on global time
                    grid_horizontal.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'grid_horizontal.stopped')
                    # update status
                    grid_horizontal.status = FINISHED
                    grid_horizontal.setAutoDraw(False)
            
            # *grid_vertical* updates
            
            # if grid_vertical is starting this frame...
            if grid_vertical.status == NOT_STARTED and tThisFlip >= 1.0-frameTolerance:
                # keep track of start time/frame for later
                grid_vertical.frameNStart = frameN  # exact frame index
                grid_vertical.tStart = t  # local t and not account for scr refresh
                grid_vertical.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(grid_vertical, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'grid_vertical.started')
                # update status
                grid_vertical.status = STARTED
                grid_vertical.setAutoDraw(True)
            
            # if grid_vertical is active this frame...
            if grid_vertical.status == STARTED:
                # update params
                pass
            
            # if grid_vertical is stopping this frame...
            if grid_vertical.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > grid_vertical.tStartRefresh + 4.1-frameTolerance:
                    # keep track of stop time/frame for later
                    grid_vertical.tStop = t  # not accounting for scr refresh
                    grid_vertical.tStopRefresh = tThisFlipGlobal  # on global time
                    grid_vertical.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'grid_vertical.stopped')
                    # update status
                    grid_vertical.status = FINISHED
                    grid_vertical.setAutoDraw(False)
            
            # *grid_center* updates
            
            # if grid_center is starting this frame...
            if grid_center.status == NOT_STARTED and tThisFlip >= 1.0-frameTolerance:
                # keep track of start time/frame for later
                grid_center.frameNStart = frameN  # exact frame index
                grid_center.tStart = t  # local t and not account for scr refresh
                grid_center.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(grid_center, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'grid_center.started')
                # update status
                grid_center.status = STARTED
                grid_center.setAutoDraw(True)
            
            # if grid_center is active this frame...
            if grid_center.status == STARTED:
                # update params
                pass
            
            # if grid_center is stopping this frame...
            if grid_center.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > grid_center.tStartRefresh + 4.1-frameTolerance:
                    # keep track of stop time/frame for later
                    grid_center.tStop = t  # not accounting for scr refresh
                    grid_center.tStopRefresh = tThisFlipGlobal  # on global time
                    grid_center.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'grid_center.stopped')
                    # update status
                    grid_center.status = FINISHED
                    grid_center.setAutoDraw(False)
            
            # *image_1* updates
            
            # if image_1 is starting this frame...
            if image_1.status == NOT_STARTED and tThisFlip >= 2.0-frameTolerance:
                # keep track of start time/frame for later
                image_1.frameNStart = frameN  # exact frame index
                image_1.tStart = t  # local t and not account for scr refresh
                image_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_1.started')
                # update status
                image_1.status = STARTED
                image_1.setAutoDraw(True)
            
            # if image_1 is active this frame...
            if image_1.status == STARTED:
                # update params
                pass
            
            # if image_1 is stopping this frame...
            if image_1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image_1.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    image_1.tStop = t  # not accounting for scr refresh
                    image_1.tStopRefresh = tThisFlipGlobal  # on global time
                    image_1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_1.stopped')
                    # update status
                    image_1.status = FINISHED
                    image_1.setAutoDraw(False)
            
            # *image_2* updates
            
            # if image_2 is starting this frame...
            if image_2.status == NOT_STARTED and tThisFlip >= 3.05-frameTolerance:
                # keep track of start time/frame for later
                image_2.frameNStart = frameN  # exact frame index
                image_2.tStart = t  # local t and not account for scr refresh
                image_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_2.started')
                # update status
                image_2.status = STARTED
                image_2.setAutoDraw(True)
            
            # if image_2 is active this frame...
            if image_2.status == STARTED:
                # update params
                pass
            
            # if image_2 is stopping this frame...
            if image_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image_2.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    image_2.tStop = t  # not accounting for scr refresh
                    image_2.tStopRefresh = tThisFlipGlobal  # on global time
                    image_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_2.stopped')
                    # update status
                    image_2.status = FINISHED
                    image_2.setAutoDraw(False)
            
            # *image_3* updates
            
            # if image_3 is starting this frame...
            if image_3.status == NOT_STARTED and tThisFlip >= 4.1-frameTolerance:
                # keep track of start time/frame for later
                image_3.frameNStart = frameN  # exact frame index
                image_3.tStart = t  # local t and not account for scr refresh
                image_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_3.started')
                # update status
                image_3.status = STARTED
                image_3.setAutoDraw(True)
            
            # if image_3 is active this frame...
            if image_3.status == STARTED:
                # update params
                pass
            
            # if image_3 is stopping this frame...
            if image_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image_3.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    image_3.tStop = t  # not accounting for scr refresh
                    image_3.tStopRefresh = tThisFlipGlobal  # on global time
                    image_3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_3.stopped')
                    # update status
                    image_3.status = FINISHED
                    image_3.setAutoDraw(False)
            
            # *text_fixation* updates
            
            # if text_fixation is starting this frame...
            if text_fixation.status == NOT_STARTED and tThisFlip >= 5.15-frameTolerance:
                # keep track of start time/frame for later
                text_fixation.frameNStart = frameN  # exact frame index
                text_fixation.tStart = t  # local t and not account for scr refresh
                text_fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_fixation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_fixation.started')
                # update status
                text_fixation.status = STARTED
                text_fixation.setAutoDraw(True)
            
            # if text_fixation is active this frame...
            if text_fixation.status == STARTED:
                # update params
                pass
            
            # if text_fixation is stopping this frame...
            if text_fixation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_fixation.tStartRefresh + 8.0-frameTolerance:
                    # keep track of stop time/frame for later
                    text_fixation.tStop = t  # not accounting for scr refresh
                    text_fixation.tStopRefresh = tThisFlipGlobal  # on global time
                    text_fixation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_fixation.stopped')
                    # update status
                    text_fixation.status = FINISHED
                    text_fixation.setAutoDraw(False)
            
            # *text_prompt* updates
            
            # if text_prompt is starting this frame...
            if text_prompt.status == NOT_STARTED and tThisFlip >= 13.15-frameTolerance:
                # keep track of start time/frame for later
                text_prompt.frameNStart = frameN  # exact frame index
                text_prompt.tStart = t  # local t and not account for scr refresh
                text_prompt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_prompt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_prompt.started')
                # update status
                text_prompt.status = STARTED
                text_prompt.setAutoDraw(True)
            
            # if text_prompt is active this frame...
            if text_prompt.status == STARTED:
                # update params
                pass
            
            # if text_prompt is stopping this frame...
            if text_prompt.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_prompt.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    text_prompt.tStop = t  # not accounting for scr refresh
                    text_prompt.tStopRefresh = tThisFlipGlobal  # on global time
                    text_prompt.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_prompt.stopped')
                    # update status
                    text_prompt.status = FINISHED
                    text_prompt.setAutoDraw(False)
            
            # *grid_outer_test* updates
            
            # if grid_outer_test is starting this frame...
            if grid_outer_test.status == NOT_STARTED and tThisFlip >= 15.15-frameTolerance:
                # keep track of start time/frame for later
                grid_outer_test.frameNStart = frameN  # exact frame index
                grid_outer_test.tStart = t  # local t and not account for scr refresh
                grid_outer_test.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(grid_outer_test, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'grid_outer_test.started')
                # update status
                grid_outer_test.status = STARTED
                grid_outer_test.setAutoDraw(True)
            
            # if grid_outer_test is active this frame...
            if grid_outer_test.status == STARTED:
                # update params
                pass
            
            # if grid_outer_test is stopping this frame...
            if grid_outer_test.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > grid_outer_test.tStartRefresh + 4.0-frameTolerance:
                    # keep track of stop time/frame for later
                    grid_outer_test.tStop = t  # not accounting for scr refresh
                    grid_outer_test.tStopRefresh = tThisFlipGlobal  # on global time
                    grid_outer_test.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'grid_outer_test.stopped')
                    # update status
                    grid_outer_test.status = FINISHED
                    grid_outer_test.setAutoDraw(False)
            
            # *grid_horizontal_test* updates
            
            # if grid_horizontal_test is starting this frame...
            if grid_horizontal_test.status == NOT_STARTED and tThisFlip >= 15.15-frameTolerance:
                # keep track of start time/frame for later
                grid_horizontal_test.frameNStart = frameN  # exact frame index
                grid_horizontal_test.tStart = t  # local t and not account for scr refresh
                grid_horizontal_test.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(grid_horizontal_test, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'grid_horizontal_test.started')
                # update status
                grid_horizontal_test.status = STARTED
                grid_horizontal_test.setAutoDraw(True)
            
            # if grid_horizontal_test is active this frame...
            if grid_horizontal_test.status == STARTED:
                # update params
                pass
            
            # if grid_horizontal_test is stopping this frame...
            if grid_horizontal_test.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > grid_horizontal_test.tStartRefresh + 4.0-frameTolerance:
                    # keep track of stop time/frame for later
                    grid_horizontal_test.tStop = t  # not accounting for scr refresh
                    grid_horizontal_test.tStopRefresh = tThisFlipGlobal  # on global time
                    grid_horizontal_test.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'grid_horizontal_test.stopped')
                    # update status
                    grid_horizontal_test.status = FINISHED
                    grid_horizontal_test.setAutoDraw(False)
            
            # *grid_vertical_test* updates
            
            # if grid_vertical_test is starting this frame...
            if grid_vertical_test.status == NOT_STARTED and tThisFlip >= 15.15-frameTolerance:
                # keep track of start time/frame for later
                grid_vertical_test.frameNStart = frameN  # exact frame index
                grid_vertical_test.tStart = t  # local t and not account for scr refresh
                grid_vertical_test.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(grid_vertical_test, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'grid_vertical_test.started')
                # update status
                grid_vertical_test.status = STARTED
                grid_vertical_test.setAutoDraw(True)
            
            # if grid_vertical_test is active this frame...
            if grid_vertical_test.status == STARTED:
                # update params
                pass
            
            # if grid_vertical_test is stopping this frame...
            if grid_vertical_test.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > grid_vertical_test.tStartRefresh + 4.0-frameTolerance:
                    # keep track of stop time/frame for later
                    grid_vertical_test.tStop = t  # not accounting for scr refresh
                    grid_vertical_test.tStopRefresh = tThisFlipGlobal  # on global time
                    grid_vertical_test.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'grid_vertical_test.stopped')
                    # update status
                    grid_vertical_test.status = FINISHED
                    grid_vertical_test.setAutoDraw(False)
            
            # *grid_center_test* updates
            
            # if grid_center_test is starting this frame...
            if grid_center_test.status == NOT_STARTED and tThisFlip >= 15.15-frameTolerance:
                # keep track of start time/frame for later
                grid_center_test.frameNStart = frameN  # exact frame index
                grid_center_test.tStart = t  # local t and not account for scr refresh
                grid_center_test.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(grid_center_test, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'grid_center_test.started')
                # update status
                grid_center_test.status = STARTED
                grid_center_test.setAutoDraw(True)
            
            # if grid_center_test is active this frame...
            if grid_center_test.status == STARTED:
                # update params
                pass
            
            # if grid_center_test is stopping this frame...
            if grid_center_test.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > grid_center_test.tStartRefresh + 4.0-frameTolerance:
                    # keep track of stop time/frame for later
                    grid_center_test.tStop = t  # not accounting for scr refresh
                    grid_center_test.tStopRefresh = tThisFlipGlobal  # on global time
                    grid_center_test.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'grid_center_test.stopped')
                    # update status
                    grid_center_test.status = FINISHED
                    grid_center_test.setAutoDraw(False)
            
            # *image_test* updates
            
            # if image_test is starting this frame...
            if image_test.status == NOT_STARTED and tThisFlip >= 15.15-frameTolerance:
                # keep track of start time/frame for later
                image_test.frameNStart = frameN  # exact frame index
                image_test.tStart = t  # local t and not account for scr refresh
                image_test.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_test, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_test.started')
                # update status
                image_test.status = STARTED
                image_test.setAutoDraw(True)
            
            # if image_test is active this frame...
            if image_test.status == STARTED:
                # update params
                pass
            
            # if image_test is stopping this frame...
            if image_test.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image_test.tStartRefresh + 4.0-frameTolerance:
                    # keep track of stop time/frame for later
                    image_test.tStop = t  # not accounting for scr refresh
                    image_test.tStopRefresh = tThisFlipGlobal  # on global time
                    image_test.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_test.stopped')
                    # update status
                    image_test.status = FINISHED
                    image_test.setAutoDraw(False)
            
            # *key_response_test* updates
            waitOnFlip = False
            
            # if key_response_test is starting this frame...
            if key_response_test.status == NOT_STARTED and tThisFlip >= 15.15-frameTolerance:
                # keep track of start time/frame for later
                key_response_test.frameNStart = frameN  # exact frame index
                key_response_test.tStart = t  # local t and not account for scr refresh
                key_response_test.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_response_test, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_response_test.started')
                # update status
                key_response_test.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_response_test.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_response_test.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if key_response_test is stopping this frame...
            if key_response_test.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_response_test.tStartRefresh + 4.0-frameTolerance:
                    # keep track of stop time/frame for later
                    key_response_test.tStop = t  # not accounting for scr refresh
                    key_response_test.tStopRefresh = tThisFlipGlobal  # on global time
                    key_response_test.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_response_test.stopped')
                    # update status
                    key_response_test.status = FINISHED
                    key_response_test.status = FINISHED
            if key_response_test.status == STARTED and not waitOnFlip:
                theseKeys = key_response_test.getKeys(keyList=['1', '2'], ignoreKeys=["escape"], waitRelease=False)
                _key_response_test_allKeys.extend(theseKeys)
                if len(_key_response_test_allKeys):
                    key_response_test.keys = _key_response_test_allKeys[-1].name  # just the last key pressed
                    key_response_test.rt = _key_response_test_allKeys[-1].rt
                    key_response_test.duration = _key_response_test_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            # Run 'Each Frame' code from trigger_trial
            if background.status == STARTED and not background_trigger_started:
                win.callOnFlip(dev.activate_line, bitmask=background_start_code)
                win.callOnFlip(eyetracker.sendMessage, background_start_code)
                background_trigger_started = True
            
            if grid_outer.status == STARTED and not grid_trigger_started:
                win.callOnFlip(dev.activate_line, bitmask=grid_start_code)
                win.callOnFlip(eyetracker.sendMessage, grid_start_code)
                grid_trigger_started = True
            
            if image_1.status == STARTED and not image_1_trigger_started:
                win.callOnFlip(dev.activate_line, bitmask=image_1_start_code)
                win.callOnFlip(eyetracker.sendMessage, image_1_start_code)
                image_1_trigger_started = True
            
            if image_2.status == STARTED and not image_2_trigger_started:
                win.callOnFlip(dev.activate_line, bitmask=image_2_start_code)
                win.callOnFlip(eyetracker.sendMessage, image_2_start_code)
                image_2_trigger_started = True
            
            if image_3.status == STARTED and not image_3_trigger_started:
                win.callOnFlip(dev.activate_line, bitmask=image_3_start_code)
                win.callOnFlip(eyetracker.sendMessage, image_3_start_code)
                image_3_trigger_started = True
            
            if text_fixation.status == STARTED and not delay_trigger_started:
                win.callOnFlip(dev.activate_line, bitmask=delay_start_code)
                win.callOnFlip(eyetracker.sendMessage, delay_start_code)
                delay_trigger_started = True
            
            if text_prompt.status == STARTED and not prompt_trigger_started:
                win.callOnFlip(dev.activate_line, bitmask=prompt_start_code)
                win.callOnFlip(eyetracker.sendMessage, prompt_start_code)
                prompt_trigger_started = True
            
            if image_test.status == STARTED and not test_trigger_started:
                win.callOnFlip(dev.activate_line, bitmask=test_start_code)
                win.callOnFlip(eyetracker.sendMessage, test_start_code)
                test_trigger_started = True
            
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                trial.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trial.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial" ---
        for thisComponent in trial.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trial
        trial.tStop = globalClock.getTime(format='float')
        trial.tStopRefresh = tThisFlipGlobal
        thisExp.addData('trial.stopped', trial.tStop)
        # check responses
        if key_response_test.keys in ['', [], None]:  # No response was made
            key_response_test.keys = None
        trials.addData('key_response_test.keys',key_response_test.keys)
        if key_response_test.keys != None:  # we had a response
            trials.addData('key_response_test.rt', key_response_test.rt)
            trials.addData('key_response_test.duration', key_response_test.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if trial.maxDurationReached:
            routineTimer.addTime(-trial.maxDuration)
        elif trial.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-19.150000)
        thisExp.nextEntry()
        
    # completed n_trials repeats of 'trials'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    # get names of stimulus parameters
    if trials.trialList in ([], [None], None):
        params = []
    else:
        params = trials.trialList[0].keys()
    # save data for this loop
    trials.saveAsText(filename + 'trials.csv', delim=',',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
    # --- Prepare to start Routine "__end__" ---
    # create an object to store info about Routine __end__
    __end__ = data.Routine(
        name='__end__',
        components=[text_thank_you, read_thank_you],
    )
    __end__.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    read_thank_you.setSound('resource/thank_you.wav', secs=2.7, hamming=True)
    read_thank_you.setVolume(1.0, log=False)
    read_thank_you.seek(0)
    # Run 'Begin Routine' code from trigger_trial_block_end
    # End of main experiment trial block
    dev.activate_line(bitmask=block_end_code)
    eyetracker.sendMessage(block_end_code)
    # no need to wait 500ms as this routine lasts 3.0s before experiment ends
    
    # store start times for __end__
    __end__.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    __end__.tStart = globalClock.getTime(format='float')
    __end__.status = STARTED
    __end__.maxDuration = None
    # keep track of which components have finished
    __end__Components = __end__.components
    for thisComponent in __end__.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "__end__" ---
    __end__.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_thank_you* updates
        
        # if text_thank_you is starting this frame...
        if text_thank_you.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_thank_you.frameNStart = frameN  # exact frame index
            text_thank_you.tStart = t  # local t and not account for scr refresh
            text_thank_you.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_thank_you, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_thank_you.status = STARTED
            text_thank_you.setAutoDraw(True)
        
        # if text_thank_you is active this frame...
        if text_thank_you.status == STARTED:
            # update params
            pass
        
        # if text_thank_you is stopping this frame...
        if text_thank_you.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_thank_you.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                text_thank_you.tStop = t  # not accounting for scr refresh
                text_thank_you.tStopRefresh = tThisFlipGlobal  # on global time
                text_thank_you.frameNStop = frameN  # exact frame index
                # update status
                text_thank_you.status = FINISHED
                text_thank_you.setAutoDraw(False)
        
        # *read_thank_you* updates
        
        # if read_thank_you is starting this frame...
        if read_thank_you.status == NOT_STARTED and t >= 0.2-frameTolerance:
            # keep track of start time/frame for later
            read_thank_you.frameNStart = frameN  # exact frame index
            read_thank_you.tStart = t  # local t and not account for scr refresh
            read_thank_you.tStartRefresh = tThisFlipGlobal  # on global time
            # update status
            read_thank_you.status = STARTED
            read_thank_you.play()  # start the sound (it finishes automatically)
        
        # if read_thank_you is stopping this frame...
        if read_thank_you.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > read_thank_you.tStartRefresh + 2.7-frameTolerance or read_thank_you.isFinished:
                # keep track of stop time/frame for later
                read_thank_you.tStop = t  # not accounting for scr refresh
                read_thank_you.tStopRefresh = tThisFlipGlobal  # on global time
                read_thank_you.frameNStop = frameN  # exact frame index
                # update status
                read_thank_you.status = FINISHED
                read_thank_you.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[read_thank_you]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            __end__.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in __end__.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "__end__" ---
    for thisComponent in __end__.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for __end__
    __end__.tStop = globalClock.getTime(format='float')
    __end__.tStopRefresh = tThisFlipGlobal
    read_thank_you.pause()  # ensure sound has stopped at end of Routine
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if __end__.maxDurationReached:
        routineTimer.addTime(-__end__.maxDuration)
    elif __end__.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    thisExp.nextEntry()
    # Run 'End Experiment' code from eeg
    # Stop EEG recording
    dev.activate_line(bitmask=127)  # trigger 127 will stop EEG
    
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    # log the filename of last_app_load.log
    logging.warn('target_last_app_load_log_file: ' + thisExp.dataFileName + '_last_app_load.log')
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
