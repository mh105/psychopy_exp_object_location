#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.1.5),
    on Thu Jun 20 19:45:17 2024
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

devices = pyxid2.get_xid_devices()

if devices:
    dev = devices[0]
    assert dev.device_name == 'Cedrus C-POD', "Incorrect C-POD detected."
    dev.set_pulse_duration(50)  # set pulse duration to 50ms

    # Start EEG recording
    dev.activate_line(bitmask=126)  # trigger 126 will start EEG
    core.wait(10)  # wait 10s for the EEG system to start recording

    # Marching lights test
    print("C-POD<->eego 7-bit trigger lines test...")
    for line in range(1, 8):  # raise lines 1-7 one at a time
        print("  raising line {} (bitmask {})".format(line, 2 ** (line-1)))
        dev.activate_line(lines=line)
        core.wait(0.5)  # wait 500ms between two consecutive triggers
    dev.con.set_digio_lines_to_mask(0)  # XidDevice.clear_all_lines()

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
psychopyVersion = '2024.1.5'
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
_winSize = [2560, 1440]
_loggingLevel = logging.getLevel('warning')
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # override logging level
    _loggingLevel = logging.getLevel(
        prefs.piloting['pilotLoggingLevel']
    )

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
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/alexhe/Dropbox (Personal)/Active_projects/PsychoPy/exp_object_location/object_location_lastrun.py',
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
    # this outputs to the screen, not a file
    logging.console.setLevel(_loggingLevel)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=_loggingLevel)
    
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
            winType='pyglet', allowStencil=False,
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
    win.mouseVisible = False
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
    ioConfig['eyetracker.hw.mouse.EyeTracker'] = {
        'name': 'tracker',
        'controls': {
            'move': [],
            'blink':('LEFT_BUTTON',),
            'saccade_threshold': 0.5,
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
    if deviceManager.getDevice('key_instruct_intro_1') is None:
        # initialise key_instruct_intro_1
        key_instruct_intro_1 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_instruct_intro_1',
        )
    if deviceManager.getDevice('key_instruct_intro_2') is None:
        # initialise key_instruct_intro_2
        key_instruct_intro_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_instruct_intro_2',
        )
    if deviceManager.getDevice('key_instruct_condition') is None:
        # initialise key_instruct_condition
        key_instruct_condition = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_instruct_condition',
        )
    if deviceManager.getDevice('key_diagram_condition') is None:
        # initialise key_diagram_condition
        key_diagram_condition = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_diagram_condition',
        )
    if deviceManager.getDevice('key_instruct_review') is None:
        # initialise key_instruct_review
        key_instruct_review = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_instruct_review',
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
    if deviceManager.getDevice('key_instruct_begin') is None:
        # initialise key_instruct_begin
        key_instruct_begin = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_instruct_begin',
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
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
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
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


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
    
    # --- Initialize components for Routine "__start__" ---
    text_start = visual.TextStim(win=win, name='text_start',
        text='We are now ready to begin...',
        font='Arial',
        units='norm', pos=(0, 0), height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
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
    # novel objects appear one at a time so no need to package
    image_fn_novel_practice = image_filenames[n_objects_practice]
    
    # Main trials
    n_trials_per_condition = 12  # each condition is repeated 12 times
    n_trials = n_conditions * n_trials_per_condition  # 72 trials during main experiment
    n_objects = n_trials * n_objects_per_trial
    # trial types - cast as list in order to pop()
    trial_type_list = rng.permutation(conditions * n_trials_per_condition).tolist()
    # image locations
    image_loc_list = [rng.permutation(locations) for _ in range(n_trials)]
    # image filenames - skipping images already used for practice trials
    image_fn = image_filenames[n_objects_practice + 1:n_objects_practice + 1 + n_objects]
    # package the images into nested lists of 3 objects
    image_fn_list = [image_fn[i * n_objects_per_trial:(i + 1) * n_objects_per_trial] for i in range(n_trials)]
    # novel objects appear one at a time so no need to package - cast as list in order to pop()
    image_fn_novel_list = image_filenames[n_objects_practice + 1 + n_objects:].tolist()
    
    
    # --- Initialize components for Routine "instruct_1" ---
    text_instruct_intro_1 = visual.TextStim(win=win, name='text_instruct_intro_1',
        text='INSTRUCTIONS\n\nIn this experiment you will see a grid with 9 squares. 3 objects will appear one at a time at different locations within the grid. You will be asked to look at these objects, then after a short delay you will be tested on how well you can remember them.\n\n\nPress the spacebar to continue',
        font='Arial',
        units='norm', pos=(0, 0), height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_instruct_intro_1 = keyboard.Keyboard(deviceName='key_instruct_intro_1')
    
    # --- Initialize components for Routine "instruct_2" ---
    text_instruct_intro_2 = visual.TextStim(win=win, name='text_instruct_intro_2',
        text="INSTRUCTIONS\n\nThere will be three different types of trials for this experiment.\n\n1) 'Remember Object' trial. \n2) 'Remember Location' trial. \n3) 'Remember Object and Location' trial. \n\nWe will now explain each type of trials separately. You will see flowcharts of the different trial types, and afterwards you will have a chance to do some practice.\n\n\nPress the spacebar to continue",
        font='Arial',
        units='norm', pos=(0, 0), height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_instruct_intro_2 = keyboard.Keyboard(deviceName='key_instruct_intro_2')
    
    # --- Initialize components for Routine "instruct_condition" ---
    # Run 'Begin Experiment' code from set_instruct_content
    instruct_loop_num = 0
    
    text_instruct_condition = visual.TextStim(win=win, name='text_instruct_condition',
        text='',
        font='Arial',
        units='norm', pos=(0, 0), height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_instruct_condition = keyboard.Keyboard(deviceName='key_instruct_condition')
    
    # --- Initialize components for Routine "diagram_condition" ---
    text_response = visual.TextStim(win=win, name='text_response',
        text='',
        font='Arial',
        units='norm', pos=(0, 0.5), height=0.08, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    image_diagram = visual.ImageStim(
        win=win,
        name='image_diagram', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.2), size=(1.25714, 0.55),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=False, depth=-2.0)
    text_continue = visual.TextStim(win=win, name='text_continue',
        text='Press the spacebar to continue',
        font='Arial',
        units='norm', pos=(0, -0.85), height=0.08, wrapWidth=1.8, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    key_diagram_condition = keyboard.Keyboard(deviceName='key_diagram_condition')
    
    # --- Initialize components for Routine "instruct_review" ---
    text_instruct_review = visual.TextStim(win=win, name='text_instruct_review',
        text="REVIEW\n\nThere are 3 types of trials in this experiment:\n'Remember Object'\n'Remember Location'\n'Remember Object and Location'.\n\nNote that these 3 types of trials will be intermixed throughout the experiment. This means that you will not know whether you need to respond to the Object or the Location or both until you see the Prompt screen.\n\n\nPress the spacebar to start practice trials",
        font='Arial',
        units='norm', pos=(0, 0), height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_instruct_review = keyboard.Keyboard(deviceName='key_instruct_review')
    
    # --- Initialize components for Routine "practice_setup" ---
    # Run 'Begin Experiment' code from setup_practice_trial
    trial_counter = 0
    
    
    # --- Initialize components for Routine "trial" ---
    background = visual.Rect(
        win=win, name='background',
        width=(1, 1)[0], height=(1, 1)[1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    grid_outer = visual.Rect(
        win=win, name='grid_outer',
        width=(0.6, 0.6)[0], height=(0.6, 0.6)[1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    grid_horizontal = visual.Rect(
        win=win, name='grid_horizontal',
        width=(0.6, 0.2)[0], height=(0.6, 0.2)[1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[1.0000, 1.0000, 1.0000],
        opacity=None, depth=-2.0, interpolate=True)
    grid_vertical = visual.Rect(
        win=win, name='grid_vertical',
        width=(0.2, 0.6)[0], height=(0.2, 0.6)[1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[1.0000, 1.0000, 1.0000],
        opacity=None, depth=-3.0, interpolate=True)
    grid_center = visual.Rect(
        win=win, name='grid_center',
        width=(0.2, 0.2)[0], height=(0.2, 0.2)[1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[1.0000, 1.0000, 1.0000],
        opacity=None, depth=-4.0, interpolate=True)
    image_1 = visual.ImageStim(
        win=win,
        name='image_1', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=False, depth=-5.0)
    image_2 = visual.ImageStim(
        win=win,
        name='image_2', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=False, depth=-6.0)
    image_3 = visual.ImageStim(
        win=win,
        name='image_3', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=False, depth=-7.0)
    text_fixation = visual.TextStim(win=win, name='text_fixation',
        text='+',
        font='Arial',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    text_prompt = visual.TextStim(win=win, name='text_prompt',
        text='',
        font='Arial',
        units='norm', pos=(0, 0), height=0.1, wrapWidth=1.8, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-9.0);
    grid_outer_test = visual.Rect(
        win=win, name='grid_outer_test',
        width=(0.6, 0.6)[0], height=(0.6, 0.6)[1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor='white',
        opacity=None, depth=-10.0, interpolate=True)
    grid_horizontal_test = visual.Rect(
        win=win, name='grid_horizontal_test',
        width=(0.6, 0.2)[0], height=(0.6, 0.2)[1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[1.0000, 1.0000, 1.0000],
        opacity=None, depth=-11.0, interpolate=True)
    grid_vertical_test = visual.Rect(
        win=win, name='grid_vertical_test',
        width=(0.2, 0.6)[0], height=(0.2, 0.6)[1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[1.0000, 1.0000, 1.0000],
        opacity=None, depth=-12.0, interpolate=True)
    grid_center_test = visual.Rect(
        win=win, name='grid_center_test',
        width=(0.2, 0.2)[0], height=(0.2, 0.2)[1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[1.0000, 1.0000, 1.0000],
        opacity=None, depth=-13.0, interpolate=True)
    image_test = visual.ImageStim(
        win=win,
        name='image_test', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=False, depth=-14.0)
    key_response_test = keyboard.Keyboard(deviceName='key_response_test')
    # Run 'Begin Experiment' code from adjust_image_size
    def scale_to_size(image_object, max_size):
        image_object.size *= max_size / max(image_object.size)  # scale to max size
        image_object._requestedSize = None  # reset for next image original size
    
    text_debug_only = visual.TextStim(win=win, name='text_debug_only',
        text='',
        font='Arial',
        units='norm', pos=(0, 0.9), height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-17.0);
    
    # --- Initialize components for Routine "practice_feedback" ---
    background_feedback = visual.Rect(
        win=win, name='background_feedback',
        width=(1, 1)[0], height=(1, 1)[1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    text_feedback = visual.TextStim(win=win, name='text_feedback',
        text='',
        font='Arial',
        units='norm', pos=(0, 0), height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "practice_checkpoint" ---
    text_checkpoint = visual.TextStim(win=win, name='text_checkpoint',
        text='Please give us a moment to check whether practice trials need to be repeated...',
        font='Arial',
        units='norm', pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_checkpoint = keyboard.Keyboard(deviceName='key_checkpoint')
    
    # --- Initialize components for Routine "instruct_begin" ---
    text_instruct_begin = visual.TextStim(win=win, name='text_instruct_begin',
        text="Great job! We will now begin the experiment.\n\nAs a reminder, the trials will ask you to remember the object, the location, or the object and its location.\n\nPress the 'Y' and 'N' keys accordingly depending on whether you think the object, location, or the object and its location are the same as shown in the 3-object sequence each time.\n\n\nPress the spacebar to start",
        font='Arial',
        units='norm', pos=(0, 0), height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_instruct_begin = keyboard.Keyboard(deviceName='key_instruct_begin')
    
    # --- Initialize components for Routine "trial_setup" ---
    
    # --- Initialize components for Routine "trial" ---
    background = visual.Rect(
        win=win, name='background',
        width=(1, 1)[0], height=(1, 1)[1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    grid_outer = visual.Rect(
        win=win, name='grid_outer',
        width=(0.6, 0.6)[0], height=(0.6, 0.6)[1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    grid_horizontal = visual.Rect(
        win=win, name='grid_horizontal',
        width=(0.6, 0.2)[0], height=(0.6, 0.2)[1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[1.0000, 1.0000, 1.0000],
        opacity=None, depth=-2.0, interpolate=True)
    grid_vertical = visual.Rect(
        win=win, name='grid_vertical',
        width=(0.2, 0.6)[0], height=(0.2, 0.6)[1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[1.0000, 1.0000, 1.0000],
        opacity=None, depth=-3.0, interpolate=True)
    grid_center = visual.Rect(
        win=win, name='grid_center',
        width=(0.2, 0.2)[0], height=(0.2, 0.2)[1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[1.0000, 1.0000, 1.0000],
        opacity=None, depth=-4.0, interpolate=True)
    image_1 = visual.ImageStim(
        win=win,
        name='image_1', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=False, depth=-5.0)
    image_2 = visual.ImageStim(
        win=win,
        name='image_2', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=False, depth=-6.0)
    image_3 = visual.ImageStim(
        win=win,
        name='image_3', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=False, depth=-7.0)
    text_fixation = visual.TextStim(win=win, name='text_fixation',
        text='+',
        font='Arial',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    text_prompt = visual.TextStim(win=win, name='text_prompt',
        text='',
        font='Arial',
        units='norm', pos=(0, 0), height=0.1, wrapWidth=1.8, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-9.0);
    grid_outer_test = visual.Rect(
        win=win, name='grid_outer_test',
        width=(0.6, 0.6)[0], height=(0.6, 0.6)[1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor='white',
        opacity=None, depth=-10.0, interpolate=True)
    grid_horizontal_test = visual.Rect(
        win=win, name='grid_horizontal_test',
        width=(0.6, 0.2)[0], height=(0.6, 0.2)[1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[1.0000, 1.0000, 1.0000],
        opacity=None, depth=-11.0, interpolate=True)
    grid_vertical_test = visual.Rect(
        win=win, name='grid_vertical_test',
        width=(0.2, 0.6)[0], height=(0.2, 0.6)[1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[1.0000, 1.0000, 1.0000],
        opacity=None, depth=-12.0, interpolate=True)
    grid_center_test = visual.Rect(
        win=win, name='grid_center_test',
        width=(0.2, 0.2)[0], height=(0.2, 0.2)[1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[1.0000, 1.0000, 1.0000],
        opacity=None, depth=-13.0, interpolate=True)
    image_test = visual.ImageStim(
        win=win,
        name='image_test', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=None,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=False, depth=-14.0)
    key_response_test = keyboard.Keyboard(deviceName='key_response_test')
    # Run 'Begin Experiment' code from adjust_image_size
    def scale_to_size(image_object, max_size):
        image_object.size *= max_size / max(image_object.size)  # scale to max size
        image_object._requestedSize = None  # reset for next image original size
    
    text_debug_only = visual.TextStim(win=win, name='text_debug_only',
        text='',
        font='Arial',
        units='norm', pos=(0, 0.9), height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-17.0);
    
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
    
    # --- Prepare to start Routine "__start__" ---
    continueRoutine = True
    # update component parameters for each repeat
    # keep track of which components have finished
    __start__Components = [text_start, etRecord]
    for thisComponent in __start__Components:
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
    routineForceEnded = not continueRoutine
    while continueRoutine:
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
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in __start__Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "__start__" ---
    for thisComponent in __start__Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.nextEntry()
    # the Routine "__start__" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instruct_1" ---
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_instruct_intro_1
    key_instruct_intro_1.keys = []
    key_instruct_intro_1.rt = []
    _key_instruct_intro_1_allKeys = []
    # keep track of which components have finished
    instruct_1Components = [text_instruct_intro_1, key_instruct_intro_1]
    for thisComponent in instruct_1Components:
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
    routineForceEnded = not continueRoutine
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
            theseKeys = key_instruct_intro_1.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=True)
            _key_instruct_intro_1_allKeys.extend(theseKeys)
            if len(_key_instruct_intro_1_allKeys):
                key_instruct_intro_1.keys = _key_instruct_intro_1_allKeys[0].name  # just the first key pressed
                key_instruct_intro_1.rt = _key_instruct_intro_1_allKeys[0].rt
                key_instruct_intro_1.duration = _key_instruct_intro_1_allKeys[0].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instruct_1Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruct_1" ---
    for thisComponent in instruct_1Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # check responses
    if key_instruct_intro_1.keys in ['', [], None]:  # No response was made
        key_instruct_intro_1.keys = None
    thisExp.addData('key_instruct_intro_1.keys',key_instruct_intro_1.keys)
    if key_instruct_intro_1.keys != None:  # we had a response
        thisExp.addData('key_instruct_intro_1.rt', key_instruct_intro_1.rt)
        thisExp.addData('key_instruct_intro_1.duration', key_instruct_intro_1.duration)
    thisExp.nextEntry()
    # the Routine "instruct_1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instruct_2" ---
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_instruct_intro_2
    key_instruct_intro_2.keys = []
    key_instruct_intro_2.rt = []
    _key_instruct_intro_2_allKeys = []
    # keep track of which components have finished
    instruct_2Components = [text_instruct_intro_2, key_instruct_intro_2]
    for thisComponent in instruct_2Components:
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
    routineForceEnded = not continueRoutine
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
            theseKeys = key_instruct_intro_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=True)
            _key_instruct_intro_2_allKeys.extend(theseKeys)
            if len(_key_instruct_intro_2_allKeys):
                key_instruct_intro_2.keys = _key_instruct_intro_2_allKeys[0].name  # just the first key pressed
                key_instruct_intro_2.rt = _key_instruct_intro_2_allKeys[0].rt
                key_instruct_intro_2.duration = _key_instruct_intro_2_allKeys[0].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instruct_2Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruct_2" ---
    for thisComponent in instruct_2Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # check responses
    if key_instruct_intro_2.keys in ['', [], None]:  # No response was made
        key_instruct_intro_2.keys = None
    thisExp.addData('key_instruct_intro_2.keys',key_instruct_intro_2.keys)
    if key_instruct_intro_2.keys != None:  # we had a response
        thisExp.addData('key_instruct_intro_2.rt', key_instruct_intro_2.rt)
        thisExp.addData('key_instruct_intro_2.duration', key_instruct_intro_2.duration)
    thisExp.nextEntry()
    # the Routine "instruct_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    instruct_loop = data.TrialHandler(nReps=3.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='instruct_loop')
    thisExp.addLoop(instruct_loop)  # add the loop to the experiment
    thisInstruct_loop = instruct_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisInstruct_loop.rgb)
    if thisInstruct_loop != None:
        for paramName in thisInstruct_loop:
            globals()[paramName] = thisInstruct_loop[paramName]
    
    for thisInstruct_loop in instruct_loop:
        currentLoop = instruct_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisInstruct_loop.rgb)
        if thisInstruct_loop != None:
            for paramName in thisInstruct_loop:
                globals()[paramName] = thisInstruct_loop[paramName]
        
        # --- Prepare to start Routine "instruct_condition" ---
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from set_instruct_content
        from pprint import pprint
        pprint(thisInstruct_loop)
        
        if instruct_loop_num == 0:  # Remember Object Condition
            instruct_condition_text = "Remember Object trial: In this type of trial you need to remember the identity of the objects shown to you.\n"
            instruct_condition_text += "\n"
            instruct_condition_text += "You will see 3 objects appearing one at a time followed by an 8 second delay. "
            instruct_condition_text += "You will then be shown a prompt screen saying 'Remember Object'. "
            instruct_condition_text += "This prompt screen will be followed by a test object in the center of the grid.\n"
            instruct_condition_text += "\n"
            instruct_condition_text += "'Remember Object' tells you that you need to decide whether this test object was one of the 3-object sequence just shown to you.\n"
            instruct_condition_text += "\n\n"
            instruct_condition_text += "Press the spacebar to continue"
        
            instruct_diagram_filename = "resource/object_trial_diagram.tif"
            
            instruct_response_text = "Remember Object Trial:\n"  # This needs to be updated!
            instruct_response_text += "\n"
            instruct_response_text += "If the test object is in the right location, press the 'Y' key to indicate YES that the test object is in the same square as it was in the 3-object sequence.\n"
            instruct_response_text += "\n"
            instruct_response_text += "If the test object is NOT in the right location, press the 'N' key to indicate NO the test object is NOT in the same square as it was in the 3-object sequence."
            
        elif instruct_loop_num == 1:  # Remember Location Condition
            instruct_condition_text = "Remember Location trial: In this type of trial you need to remember the location of the objects shown to you.\n"
            instruct_condition_text += "\n"
            instruct_condition_text += "You will see 3 objects appearing one at a time followed by an 8 second delay. "
            instruct_condition_text += "You will then be shown a prompt screen saying 'Remember Location'. "
            instruct_condition_text += "This prompt screen will be followed by a dot in one of the squares of the grid.\n"
            instruct_condition_text += "\n"
            instruct_condition_text += "'Remember Location' tells you that you need to decide whether this square was previously occupied by any object in the 3-object sequence.\n"
            instruct_condition_text += "\n\n"
            instruct_condition_text += "Press the spacebar to continue"
        
            instruct_diagram_filename = "resource/location_trial_diagram.tif"
        
            instruct_response_text = "Remember Location Trial:\n"  # This needs to be updated!
            instruct_response_text += "\n"
            instruct_response_text += "If the test object is in the right location, press the 'Y' key to indicate YES that the test object is in the same square as it was in the 3-object sequence.\n"
            instruct_response_text += "\n"
            instruct_response_text += "If the test object is NOT in the right location, press the 'N' key to indicate NO the test object is NOT in the same square as it was in the 3-object sequence."
        
        elif instruct_loop_num == 2:  # Remember Object and Location Condition
            instruct_condition_text = "Remember Object and Location trial: In this type of trial you need to remember both the identity of the objects and their locations.\n"
            instruct_condition_text += "\n"
            instruct_condition_text += "As with the other trials, you will see 3 objects appearing one at a time followed by an 8 second delay. "
            instruct_condition_text += "You will then be shown a prompt screen saying 'Remember Object and Location'. "
            instruct_condition_text += "This prompt screen will be followed by a test object in one of the squares of the grid.\n"
            instruct_condition_text += "\n"
            instruct_condition_text += "'Remember Object and Location' tells you that you need to decide whether this object is in the right location. What right means is that the test object is in the same square as it was in the 3-object sequence.\n"
            instruct_condition_text += "\n\n"
            instruct_condition_text += "Press the spacebar to continue"
            
            instruct_diagram_filename = "resource/object_location_trial_diagram.tif"
            
            instruct_response_text = "Remember Object and Location Trial:\n"
            instruct_response_text += "\n"
            instruct_response_text += "If the test object is in the right location, press the 'Y' key to indicate YES that the test object is in the same square as it was in the 3-object sequence.\n"
            instruct_response_text += "\n"
            instruct_response_text += "If the test object is NOT in the right location, press the 'N' key to indicate NO the test object is NOT in the same square as it was in the 3-object sequence."
        
        instruct_loop_num += 1
        
        text_instruct_condition.setText(instruct_condition_text)
        # create starting attributes for key_instruct_condition
        key_instruct_condition.keys = []
        key_instruct_condition.rt = []
        _key_instruct_condition_allKeys = []
        # keep track of which components have finished
        instruct_conditionComponents = [text_instruct_condition, key_instruct_condition]
        for thisComponent in instruct_conditionComponents:
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
        routineForceEnded = not continueRoutine
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
                theseKeys = key_instruct_condition.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=True)
                _key_instruct_condition_allKeys.extend(theseKeys)
                if len(_key_instruct_condition_allKeys):
                    key_instruct_condition.keys = _key_instruct_condition_allKeys[0].name  # just the first key pressed
                    key_instruct_condition.rt = _key_instruct_condition_allKeys[0].rt
                    key_instruct_condition.duration = _key_instruct_condition_allKeys[0].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in instruct_conditionComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "instruct_condition" ---
        for thisComponent in instruct_conditionComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # check responses
        if key_instruct_condition.keys in ['', [], None]:  # No response was made
            key_instruct_condition.keys = None
        instruct_loop.addData('key_instruct_condition.keys',key_instruct_condition.keys)
        if key_instruct_condition.keys != None:  # we had a response
            instruct_loop.addData('key_instruct_condition.rt', key_instruct_condition.rt)
            instruct_loop.addData('key_instruct_condition.duration', key_instruct_condition.duration)
        # the Routine "instruct_condition" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "diagram_condition" ---
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
        # keep track of which components have finished
        diagram_conditionComponents = [text_response, image_diagram, text_continue, key_diagram_condition]
        for thisComponent in diagram_conditionComponents:
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
        routineForceEnded = not continueRoutine
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
                theseKeys = key_diagram_condition.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=True)
                _key_diagram_condition_allKeys.extend(theseKeys)
                if len(_key_diagram_condition_allKeys):
                    key_diagram_condition.keys = _key_diagram_condition_allKeys[0].name  # just the first key pressed
                    key_diagram_condition.rt = _key_diagram_condition_allKeys[0].rt
                    key_diagram_condition.duration = _key_diagram_condition_allKeys[0].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in diagram_conditionComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "diagram_condition" ---
        for thisComponent in diagram_conditionComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # check responses
        if key_diagram_condition.keys in ['', [], None]:  # No response was made
            key_diagram_condition.keys = None
        instruct_loop.addData('key_diagram_condition.keys',key_diagram_condition.keys)
        if key_diagram_condition.keys != None:  # we had a response
            instruct_loop.addData('key_diagram_condition.rt', key_diagram_condition.rt)
            instruct_loop.addData('key_diagram_condition.duration', key_diagram_condition.duration)
        # the Routine "diagram_condition" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
    # completed 3.0 repeats of 'instruct_loop'
    
    
    # --- Prepare to start Routine "instruct_review" ---
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_instruct_review
    key_instruct_review.keys = []
    key_instruct_review.rt = []
    _key_instruct_review_allKeys = []
    # keep track of which components have finished
    instruct_reviewComponents = [text_instruct_review, key_instruct_review]
    for thisComponent in instruct_reviewComponents:
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
    routineForceEnded = not continueRoutine
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
            theseKeys = key_instruct_review.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=True)
            _key_instruct_review_allKeys.extend(theseKeys)
            if len(_key_instruct_review_allKeys):
                key_instruct_review.keys = _key_instruct_review_allKeys[0].name  # just the first key pressed
                key_instruct_review.rt = _key_instruct_review_allKeys[0].rt
                key_instruct_review.duration = _key_instruct_review_allKeys[0].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instruct_reviewComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruct_review" ---
    for thisComponent in instruct_reviewComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # check responses
    if key_instruct_review.keys in ['', [], None]:  # No response was made
        key_instruct_review.keys = None
    thisExp.addData('key_instruct_review.keys',key_instruct_review.keys)
    if key_instruct_review.keys != None:  # we had a response
        thisExp.addData('key_instruct_review.rt', key_instruct_review.rt)
        thisExp.addData('key_instruct_review.duration', key_instruct_review.duration)
    thisExp.nextEntry()
    # the Routine "instruct_review" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    practice_loop = data.TrialHandler(nReps=99.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='practice_loop')
    thisExp.addLoop(practice_loop)  # add the loop to the experiment
    thisPractice_loop = practice_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPractice_loop.rgb)
    if thisPractice_loop != None:
        for paramName in thisPractice_loop:
            globals()[paramName] = thisPractice_loop[paramName]
    
    for thisPractice_loop in practice_loop:
        currentLoop = practice_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisPractice_loop.rgb)
        if thisPractice_loop != None:
            for paramName in thisPractice_loop:
                globals()[paramName] = thisPractice_loop[paramName]
        
        # set up handler to look after randomisation of conditions etc
        practice_trials = data.TrialHandler(nReps=n_trials_practice, method='random', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='practice_trials')
        thisExp.addLoop(practice_trials)  # add the loop to the experiment
        thisPractice_trial = practice_trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisPractice_trial.rgb)
        if thisPractice_trial != None:
            for paramName in thisPractice_trial:
                globals()[paramName] = thisPractice_trial[paramName]
        
        for thisPractice_trial in practice_trials:
            currentLoop = practice_trials
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisPractice_trial.rgb)
            if thisPractice_trial != None:
                for paramName in thisPractice_trial:
                    globals()[paramName] = thisPractice_trial[paramName]
            
            # --- Prepare to start Routine "practice_setup" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('practice_setup.started', globalClock.getTime(format='float'))
            # Run 'Begin Routine' code from setup_practice_trial
            # Obtain practice trial specific variables
            image_fn = image_fn_practice_list[trial_counter]
            image_loc = image_loc_practice_list[trial_counter]
            trial_type = trial_type_practice_list[trial_counter]
            
            # Set up prompt, filenames, location, etc.
            if trial_type == 'objsame':
                test_prompt = "Remember Object"
                image_test_fn = rng.choice(image_fn)  # one of three study objects
                image_test_loc = [0, 0]  # image location: center
                correct_resp = 'y'
                
            elif trial_type == 'objdifferent':
                test_prompt = "Remember Object"
                image_test_fn = image_fn_novel_practice  # a novel object
                image_test_loc = [0, 0]  # image location: center
                correct_resp = 'n'
                
            elif trial_type == 'locsame':
                test_prompt = "Remember Location"
                image_test_fn = image_dot_fn  # show dot image
                image_test_loc = rng.choice(image_loc[:3])  # one of shown locations
                correct_resp = 'y'
                
            elif trial_type == 'locdifferent':
                test_prompt = "Remember Location"
                image_test_fn = image_dot_fn  # show dot image
                image_test_loc = image_loc[3]  # a novel location that wasn't shown
                correct_resp = 'n'
            
            elif trial_type == 'objlocsame':
                test_prompt = "Remember Object and Location"
                i = rng.choice([0, 1, 2])  # select one of the three study objects
                image_test_fn = image_fn[i]  # show the selected object
                image_test_loc = image_loc[i]  # and its corresponding location
                correct_resp = 'y'
                
            elif trial_type == 'objlocdifferent':
                test_prompt = "Remember Object and Location"
                index = [0, 1, 2]
                i = rng.choice(index)  # select one of the three study objects
                _ = index.pop(i)  # remove the selected index from the list
                image_test_fn = image_fn[i]  # show the selected object
                image_test_loc = image_loc[rng.choice(index)]  # select from remaining locations
                correct_resp = 'n'
            
            else:
                error("Unrecognized trial type. Please double check.")
            
            # keep track of which components have finished
            practice_setupComponents = []
            for thisComponent in practice_setupComponents:
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
            routineForceEnded = not continueRoutine
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
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in practice_setupComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "practice_setup" ---
            for thisComponent in practice_setupComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('practice_setup.stopped', globalClock.getTime(format='float'))
            # Run 'End Routine' code from setup_practice_trial
            trial_counter += 1
            
            # the Routine "practice_setup" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "trial" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('trial.started', globalClock.getTime(format='float'))
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
            
            text_debug_only.setText(trial_type)
            # keep track of which components have finished
            trialComponents = [background, grid_outer, grid_horizontal, grid_vertical, grid_center, image_1, image_2, image_3, text_fixation, text_prompt, grid_outer_test, grid_horizontal_test, grid_vertical_test, grid_center_test, image_test, key_response_test, text_debug_only]
            for thisComponent in trialComponents:
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
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 18.6:
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
                    if tThisFlipGlobal > background.tStartRefresh + 18.6-frameTolerance:
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
                if grid_outer.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
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
                if grid_horizontal.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
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
                if grid_vertical.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
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
                if grid_center.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
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
                if image_1.status == NOT_STARTED and tThisFlip >= 1.5-frameTolerance:
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
                if image_2.status == NOT_STARTED and tThisFlip >= 2.55-frameTolerance:
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
                if image_3.status == NOT_STARTED and tThisFlip >= 3.6-frameTolerance:
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
                if text_fixation.status == NOT_STARTED and tThisFlip >= 4.6-frameTolerance:
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
                if text_prompt.status == NOT_STARTED and tThisFlip >= 12.6-frameTolerance:
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
                if grid_outer_test.status == NOT_STARTED and tThisFlip >= 14.6-frameTolerance:
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
                if grid_horizontal_test.status == NOT_STARTED and tThisFlip >= 14.6-frameTolerance:
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
                if grid_vertical_test.status == NOT_STARTED and tThisFlip >= 14.6-frameTolerance:
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
                if grid_center_test.status == NOT_STARTED and tThisFlip >= 14.6-frameTolerance:
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
                if image_test.status == NOT_STARTED and tThisFlip >= 14.6-frameTolerance:
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
                if key_response_test.status == NOT_STARTED and tThisFlip >= 14.6-frameTolerance:
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
                    theseKeys = key_response_test.getKeys(keyList=['y', 'n'], ignoreKeys=["escape"], waitRelease=False)
                    _key_response_test_allKeys.extend(theseKeys)
                    if len(_key_response_test_allKeys):
                        key_response_test.keys = _key_response_test_allKeys[0].name  # just the first key pressed
                        key_response_test.rt = _key_response_test_allKeys[0].rt
                        key_response_test.duration = _key_response_test_allKeys[0].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # *text_debug_only* updates
                
                # if text_debug_only is starting this frame...
                if text_debug_only.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                    # keep track of start time/frame for later
                    text_debug_only.frameNStart = frameN  # exact frame index
                    text_debug_only.tStart = t  # local t and not account for scr refresh
                    text_debug_only.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_debug_only, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_debug_only.started')
                    # update status
                    text_debug_only.status = STARTED
                    text_debug_only.setAutoDraw(True)
                
                # if text_debug_only is active this frame...
                if text_debug_only.status == STARTED:
                    # update params
                    pass
                
                # if text_debug_only is stopping this frame...
                if text_debug_only.status == STARTED:
                    # is it time to stop? (based on local clock)
                    if tThisFlip > 18.6-frameTolerance:
                        # keep track of stop time/frame for later
                        text_debug_only.tStop = t  # not accounting for scr refresh
                        text_debug_only.tStopRefresh = tThisFlipGlobal  # on global time
                        text_debug_only.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_debug_only.stopped')
                        # update status
                        text_debug_only.status = FINISHED
                        text_debug_only.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trialComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial" ---
            for thisComponent in trialComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('trial.stopped', globalClock.getTime(format='float'))
            # check responses
            if key_response_test.keys in ['', [], None]:  # No response was made
                key_response_test.keys = None
            practice_trials.addData('key_response_test.keys',key_response_test.keys)
            if key_response_test.keys != None:  # we had a response
                practice_trials.addData('key_response_test.rt', key_response_test.rt)
                practice_trials.addData('key_response_test.duration', key_response_test.duration)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-18.600000)
            
            # --- Prepare to start Routine "practice_feedback" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('practice_feedback.started', globalClock.getTime(format='float'))
            # Run 'Begin Routine' code from set_feedback_text
            if not key_response_test.keys:
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
            # keep track of which components have finished
            practice_feedbackComponents = [background_feedback, text_feedback]
            for thisComponent in practice_feedbackComponents:
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
            routineForceEnded = not continueRoutine
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
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'background_feedback.started')
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
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'background_feedback.stopped')
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
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_feedback.started')
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
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_feedback.stopped')
                        # update status
                        text_feedback.status = FINISHED
                        text_feedback.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in practice_feedbackComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "practice_feedback" ---
            for thisComponent in practice_feedbackComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('practice_feedback.stopped', globalClock.getTime(format='float'))
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-2.000000)
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed n_trials_practice repeats of 'practice_trials'
        
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
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('practice_checkpoint.started', globalClock.getTime(format='float'))
        # create starting attributes for key_checkpoint
        key_checkpoint.keys = []
        key_checkpoint.rt = []
        _key_checkpoint_allKeys = []
        # keep track of which components have finished
        practice_checkpointComponents = [text_checkpoint, key_checkpoint]
        for thisComponent in practice_checkpointComponents:
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
        routineForceEnded = not continueRoutine
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
                    key_checkpoint.keys = _key_checkpoint_allKeys[0].name  # just the first key pressed
                    key_checkpoint.rt = _key_checkpoint_allKeys[0].rt
                    key_checkpoint.duration = _key_checkpoint_allKeys[0].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in practice_checkpointComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "practice_checkpoint" ---
        for thisComponent in practice_checkpointComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('practice_checkpoint.stopped', globalClock.getTime(format='float'))
        # check responses
        if key_checkpoint.keys in ['', [], None]:  # No response was made
            key_checkpoint.keys = None
        practice_loop.addData('key_checkpoint.keys',key_checkpoint.keys)
        if key_checkpoint.keys != None:  # we had a response
            practice_loop.addData('key_checkpoint.rt', key_checkpoint.rt)
            practice_loop.addData('key_checkpoint.duration', key_checkpoint.duration)
        # Run 'End Routine' code from code_checkpoint
        if key_checkpoint.keys == 'r':  # repeat the practice trials
            # reset trial counter variable
            trial_counter = 0
        else:  # 'o' means proceed to main experiment
            practice_loop.finished = True
        
        # the Routine "practice_checkpoint" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
    # completed 99.0 repeats of 'practice_loop'
    
    
    # --- Prepare to start Routine "instruct_begin" ---
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_instruct_begin
    key_instruct_begin.keys = []
    key_instruct_begin.rt = []
    _key_instruct_begin_allKeys = []
    # keep track of which components have finished
    instruct_beginComponents = [text_instruct_begin, key_instruct_begin]
    for thisComponent in instruct_beginComponents:
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
    routineForceEnded = not continueRoutine
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
            theseKeys = key_instruct_begin.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=True)
            _key_instruct_begin_allKeys.extend(theseKeys)
            if len(_key_instruct_begin_allKeys):
                key_instruct_begin.keys = _key_instruct_begin_allKeys[0].name  # just the first key pressed
                key_instruct_begin.rt = _key_instruct_begin_allKeys[0].rt
                key_instruct_begin.duration = _key_instruct_begin_allKeys[0].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instruct_beginComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruct_begin" ---
    for thisComponent in instruct_beginComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # check responses
    if key_instruct_begin.keys in ['', [], None]:  # No response was made
        key_instruct_begin.keys = None
    thisExp.addData('key_instruct_begin.keys',key_instruct_begin.keys)
    if key_instruct_begin.keys != None:  # we had a response
        thisExp.addData('key_instruct_begin.rt', key_instruct_begin.rt)
        thisExp.addData('key_instruct_begin.duration', key_instruct_begin.duration)
    thisExp.nextEntry()
    # the Routine "instruct_begin" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler(nReps=n_trials, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='trials')
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "trial_setup" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('trial_setup.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from setup_image_trial
        # Obtain trial specific variables
        image_fn = image_fn_list.pop()
        image_loc = image_loc_list.pop()
        trial_type = trial_type_list.pop()
        
        # Set up prompt, filenames, location, etc.
        if trial_type == 'objsame':
            test_prompt = "Remember Object"
            image_test_fn = rng.choice(image_fn)  # one of three study objects
            image_test_loc = [0, 0]  # image location: center
            
        elif trial_type == 'objdifferent':
            test_prompt = "Remember Object"
            image_test_fn = image_fn_novel_list.pop()  # a novel object
            image_test_loc = [0, 0]  # image location: center
            
        elif trial_type == 'locsame':
            test_prompt = "Remember Location"
            image_test_fn = image_dot_fn  # show dot image
            image_test_loc = rng.choice(image_loc[:3])  # one of shown locations
            
        elif trial_type == 'locdifferent':
            test_prompt = "Remember Location"
            image_test_fn = image_dot_fn  # show dot image
            image_test_loc = image_loc[3]  # a novel location that wasn't shown
        
        elif trial_type == 'objlocsame':
            test_prompt = "Remember Object and Location"
            i = rng.choice([0, 1, 2])  # select one of the three study objects
            image_test_fn = image_fn[i]  # show the selected object
            image_test_loc = image_loc[i]  # and its corresponding location
            
        elif trial_type == 'objlocdifferent':
            test_prompt = "Remember Object and Location"
            index = [0, 1, 2]
            i = rng.choice(index)  # select one of the three study objects
            _ = index.pop(i)  # remove the selected index from the list
            image_test_fn = image_fn[i]  # show the selected object
            image_test_loc = image_loc[rng.choice(index)]  # select from remaining locations
        
        else:
            error("Unrecognized trial type. Please double check.")
        
        # keep track of which components have finished
        trial_setupComponents = []
        for thisComponent in trial_setupComponents:
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
        routineForceEnded = not continueRoutine
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
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trial_setupComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial_setup" ---
        for thisComponent in trial_setupComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('trial_setup.stopped', globalClock.getTime(format='float'))
        # the Routine "trial_setup" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "trial" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('trial.started', globalClock.getTime(format='float'))
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
        
        text_debug_only.setText(trial_type)
        # keep track of which components have finished
        trialComponents = [background, grid_outer, grid_horizontal, grid_vertical, grid_center, image_1, image_2, image_3, text_fixation, text_prompt, grid_outer_test, grid_horizontal_test, grid_vertical_test, grid_center_test, image_test, key_response_test, text_debug_only]
        for thisComponent in trialComponents:
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
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 18.6:
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
                if tThisFlipGlobal > background.tStartRefresh + 18.6-frameTolerance:
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
            if grid_outer.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
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
            if grid_horizontal.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
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
            if grid_vertical.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
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
            if grid_center.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
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
            if image_1.status == NOT_STARTED and tThisFlip >= 1.5-frameTolerance:
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
            if image_2.status == NOT_STARTED and tThisFlip >= 2.55-frameTolerance:
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
            if image_3.status == NOT_STARTED and tThisFlip >= 3.6-frameTolerance:
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
            if text_fixation.status == NOT_STARTED and tThisFlip >= 4.6-frameTolerance:
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
            if text_prompt.status == NOT_STARTED and tThisFlip >= 12.6-frameTolerance:
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
            if grid_outer_test.status == NOT_STARTED and tThisFlip >= 14.6-frameTolerance:
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
            if grid_horizontal_test.status == NOT_STARTED and tThisFlip >= 14.6-frameTolerance:
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
            if grid_vertical_test.status == NOT_STARTED and tThisFlip >= 14.6-frameTolerance:
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
            if grid_center_test.status == NOT_STARTED and tThisFlip >= 14.6-frameTolerance:
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
            if image_test.status == NOT_STARTED and tThisFlip >= 14.6-frameTolerance:
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
            if key_response_test.status == NOT_STARTED and tThisFlip >= 14.6-frameTolerance:
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
                theseKeys = key_response_test.getKeys(keyList=['y', 'n'], ignoreKeys=["escape"], waitRelease=False)
                _key_response_test_allKeys.extend(theseKeys)
                if len(_key_response_test_allKeys):
                    key_response_test.keys = _key_response_test_allKeys[0].name  # just the first key pressed
                    key_response_test.rt = _key_response_test_allKeys[0].rt
                    key_response_test.duration = _key_response_test_allKeys[0].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *text_debug_only* updates
            
            # if text_debug_only is starting this frame...
            if text_debug_only.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_debug_only.frameNStart = frameN  # exact frame index
                text_debug_only.tStart = t  # local t and not account for scr refresh
                text_debug_only.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_debug_only, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_debug_only.started')
                # update status
                text_debug_only.status = STARTED
                text_debug_only.setAutoDraw(True)
            
            # if text_debug_only is active this frame...
            if text_debug_only.status == STARTED:
                # update params
                pass
            
            # if text_debug_only is stopping this frame...
            if text_debug_only.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > 18.6-frameTolerance:
                    # keep track of stop time/frame for later
                    text_debug_only.tStop = t  # not accounting for scr refresh
                    text_debug_only.tStopRefresh = tThisFlipGlobal  # on global time
                    text_debug_only.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_debug_only.stopped')
                    # update status
                    text_debug_only.status = FINISHED
                    text_debug_only.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trialComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial" ---
        for thisComponent in trialComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('trial.stopped', globalClock.getTime(format='float'))
        # check responses
        if key_response_test.keys in ['', [], None]:  # No response was made
            key_response_test.keys = None
        trials.addData('key_response_test.keys',key_response_test.keys)
        if key_response_test.keys != None:  # we had a response
            trials.addData('key_response_test.rt', key_response_test.rt)
            trials.addData('key_response_test.duration', key_response_test.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-18.600000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed n_trials repeats of 'trials'
    
    # get names of stimulus parameters
    if trials.trialList in ([], [None], None):
        params = []
    else:
        params = trials.trialList[0].keys()
    # save data for this loop
    trials.saveAsText(filename + 'trials.csv', delim=',',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
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
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
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
