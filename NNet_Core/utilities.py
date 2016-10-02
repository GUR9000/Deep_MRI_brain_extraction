"""
This software is an implementation of

Deep MRI brain extraction: A 3D convolutional neural network for skull stripping

You can download the paper at http://dx.doi.org/10.1016/j.neuroimage.2016.01.024

If you use this software for your projects please cite:

Kleesiek and Urban et al, Deep MRI brain extraction: A 3D convolutional neural network for skull stripping,
NeuroImage, Volume 129, April 2016, Pages 460-469.

The MIT License (MIT)

Copyright (c) 2016 Gregor Urban, Jens Kleesiek

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import time
import numpy
import numpy as np


import os
from os import makedirs as _makedirs
from os.path import exists as _exists


def extract_filename(string, remove_trailing_ftype=True, trailing_type_max_len=7):
    """ removes path (in front of the file name) and removes the file-type after the '.' (optional).
    returns: path & file_name"""
    A = string.replace("\\","/").split("/")
    path = ("/".join(A[:-1]))+"/"
    if len(path)==1:
        path=""
    B=A[-1]
    if remove_trailing_ftype:
        file_name = ".".join(B.split(".")[:-1])
        if len(file_name)==0 or len(B)-len(file_name)>(trailing_type_max_len+1):
            file_name = B
    else:
        file_name=B
    return path, file_name

def mkdir(path):
    if len(path)>1 and _exists(path)==0:
        _makedirs(path)



def load_text(fname, SplitLines=False):
    try:
        f=open(fname,'r')
    except:
        print("load_text:: cannot open file <",fname,">")
        return None
    text = f.readlines()
    f.close()
    if SplitLines:
        return "".join(text).split('\n')
    else:
        return "".join(text)



class Logger():
    def __init__(self, save_file = "defaul_log.txt", save_every_n_minutes = 10, log_additional_info = True):
        self._save_file = save_file
        path = extract_filename(self._save_file)[0]
        if path!="" and os.path.exists(path)==False:
            mkdir(path)
        self._logf = open(self._save_file, "a+")
        self._save_every_x_seconds = save_every_n_minutes*60.
        self._log_additional_info = log_additional_info

        if self._log_additional_info:
            timestamp = time.strftime("%m/%d/%Y %H:%M:%S\n", time.localtime(time.time()))
            self._logf.write( "\n"*5 )
            self._logf.write( "LOG START\n" )
            self._logf.write( timestamp+'\n\n' )

        self._logf.flush()
        self._last_flush = time.time()

        self._alive = True

    def log(self, string, append_endline = True):
        assert self._alive
        if not isinstance(string,str):
            string=str(string)
        try:
            if append_endline and string[-1]!= "\n":
                string += "\n"
            self._logf.write( string )

            if (time.time() - self._last_flush) > self._save_every_x_seconds:
                self._logf.flush()
                self._last_flush = time.time()
        except:
            print("STRING IS IVALID! <",string,'>')

    def log_and_print(self, string_or_list):
        """If input is a list, then <" ".join(string_or_list)> will be printed and logged to the file. The elements must not be strings (this function does transform it if necessary)"""
        if isinstance(string_or_list,list):
            string_or_list = " ".join([str(x) for x in string_or_list])
        print(string_or_list)
        self.log(string_or_list)


    def close(self):
        if self._alive == True:
            if self._log_additional_info:
                self._logf.write( "\n"*2)
                self._logf.write( "LOG END\n" )
                timestamp = time.strftime("%m/%d/%Y %H:%M:%S\n", time.localtime(time.time()))
                self._logf.write( timestamp+'\n\n' )
            self._logf.close()
            self._alive = False

    def __del__(self):
        try:
            self.close()
        except:
            pass



class LR_scheduler(object):
    def __init__(self,
                 cnn,
                 max_training_time_minutes = None,
                 max_training_steps = None,
                 LR_start = 1e-3,
                 automated_LR_scaling_enabled = True,
                 automated_LR_scaling_magnitude = 0.5,
                 automated_LR_scaling_wait_steps = 10,
                 automated_LR_scaling_max_LR_reduction_factor_before_termination = 1000,
                 automated_LR_scaling_minimum_n_steps_for_subsequent_reduction = 6,
                 automated_kill_after_n_unchanged_steps = 30,
                 automated_kill_if_bad_enabled                   = False,
                 automated_kill_if_bad__time_of_decision_minutes = 10,
                 automated_kill_if_bad__killscore                = 0.1
                 ):
        """


        automated_LR_scaling_magnitude = 0.5:
            usually: no need to change this!

        automated_LR_scaling_wait_steps = 10:
            if not improving for this many steps/epochs, LR will be reduced

        automated_LR_scaling_max_LR_reduction_factor_before_termination = 1000:
            if LR was reduced by this amount, then training stops

        automated_kill_after_n_unchanged_steps = 30:
            always active: if LR reductions don't help to improve on the training/validation set, then training stops

        automated_kill_if_bad_enabled  = False:
            compares 'score' (accuracy/ negative loss) to <automated_kill_if_bad__killscore> after X minutes, if not HIGHER (i.e. better), then training is stopped.

        max_training_time_minutes:
            Training is stopped after this time (None: never terminate due to spent time)

        max_training_steps:
            Training is stopped after this many steps (None: never terminate due to number of steps taken). A step might be a single SGD update or a full epoch, the actual granularity is controlled by calls to the .tick() function/method.

        returns:
        -------

        nothing
        """
        self._cnn = cnn
        self._LR_start = LR_start
        self._max_training_steps = max_training_steps
        self._max_training_time_minutes = max_training_time_minutes

        assert automated_LR_scaling_magnitude>0 and automated_LR_scaling_magnitude < 1.
        assert automated_LR_scaling_wait_steps > 1
        assert automated_LR_scaling_max_LR_reduction_factor_before_termination > 10
        assert automated_LR_scaling_minimum_n_steps_for_subsequent_reduction < automated_LR_scaling_wait_steps
        assert automated_kill_after_n_unchanged_steps > 1.5 * automated_LR_scaling_wait_steps

        self._automated_LR_scaling_enabled                                    = automated_LR_scaling_enabled
        self._automated_LR_scaling_magnitude                                  = automated_LR_scaling_magnitude
        self._automated_LR_scaling_wait_steps                                 = automated_LR_scaling_wait_steps
        self._automated_LR_scaling_max_LR_reduction_factor_before_termination = automated_LR_scaling_max_LR_reduction_factor_before_termination
        self._automated_LR_scaling_minimum_n_steps_for_subsequent_reduction   = automated_LR_scaling_minimum_n_steps_for_subsequent_reduction

        self._automated_kill_after_n_unchanged_steps          = automated_kill_after_n_unchanged_steps

        self._automated_kill_if_bad_enabled                   = automated_kill_if_bad_enabled
        self._automated_kill_if_bad__time_of_decision_minutes = automated_kill_if_bad__time_of_decision_minutes
        self._automated_kill_if_bad__killscore                = automated_kill_if_bad__killscore

        self._automated_LR_reduction__best_score_at_step = 0
        self._automated_LR_reduction__best_score = -9e9
        self._automated_LR_reduction_steps_so_far = 0
        self._automated_LR_scaling_last_step_active = 0

        self._training_start_time = time.clock()
        self._cnn.set_SGD_LR(self._LR_start)
        #######################################################################



    def tick(self, current_step, current_score):
        """
        updates LR if appropriate.

        current_step:
            current epoch (index/number) OR number of SGD updates completed OR anything similar that is useful.

        current_score:
            HIGHER means BETTER. Thus, if you have a loss value (NOT accuracy or similar), then just pass the negative value to this function instead.

        returns:
        -------

        terminate_training (boolean)

        """

#        print 'current_step', current_step,'current_score', current_score

        terminate_training = False

        if self._max_training_steps is not None:
            if (current_step >= self._max_training_steps):
                print self,":: Training stopped as the iteration limit was reached!"
                terminate_training = True

        if self._max_training_time_minutes is not None:
            if (time.clock() - self._training_start_time)/60. > self._max_training_time_minutes:
                print self,":: Training stopped as the time limit was reached!"
                terminate_training = True

        if self._automated_LR_scaling_enabled:
            if current_score > self._automated_LR_reduction__best_score:
                self._automated_LR_reduction__best_score = current_score
                self._automated_LR_reduction__best_score_at_step = current_step
            else:
                if (current_step - self._automated_LR_reduction__best_score_at_step)> self._automated_LR_scaling_wait_steps and (current_step - self._automated_LR_scaling_last_step_active) >= self._automated_LR_scaling_minimum_n_steps_for_subsequent_reduction:
                    old = self._cnn.get_SGD_LR()
                    self._automated_LR_scaling_last_step_active = current_step
                    self._cnn.set_SGD_LR( np.float32( self._automated_LR_scaling_magnitude * old))
                    print self,":: AUTOMATED LR-control: Reducing LR:  old value was "+str(old)+" new value is "+str(self._cnn.get_SGD_LR())
                    self._automated_LR_reduction_steps_so_far += 1

                    if 1./(self._automated_LR_scaling_magnitude**self._automated_LR_reduction_steps_so_far) >= self._automated_LR_scaling_max_LR_reduction_factor_before_termination:
                        terminate_training = True
                        print self,":: AUTOMATED LR-control: Terminating training as maximum reduction rate was reached."

        if self._automated_kill_if_bad_enabled and time.clock() - self._training_start_time > self._automated_kill_if_bad__time_of_decision_minutes*60:
            if current_score < self._automated_kill_if_bad__killscore:
                print self,":: AUTOMATED LR-control: Terminating training as score is too low."
                terminate_training = True

        # early kill
        if self._automated_kill_after_n_unchanged_steps is not None:
            if (current_step - self._automated_LR_reduction__best_score_at_step) > self._automated_kill_after_n_unchanged_steps:
                terminate_training = True
                print self,":: AUTOMATED Kill-control: Terminating training as unchanged accuracy for "+str(self._automated_kill_after_n_unchanged_steps)+" steps."

#        print self.__dict__

        return terminate_training
        #######################################################################

###############################################################################

class AutosaveControl:
    def __init__(self, cnn, training_time_minutes = 120, LR_start = 9e-5, LR_end = 5e-6, save_name="auto",
                 autosave_n_files = 10, exponential_interpolation = True, autosave_frequency_minutes = 60, save_path=""):
        self.autosave_frequency = autosave_frequency_minutes #save all <autosave_frequency_minutes> minutes
        self.training_start_time = time.clock()
        if len(save_path):
            save_path = save_path.replace('\\','/')
            if save_path[-1]!='/':
                save_path+='/'
            save_name = save_path + save_name
        self.save_name = save_name
        self.training_termination_time = 60.0 * training_time_minutes # training ends XXX seconds after start
        self.status_printing_interval = 20 #seconds, updates LR too if update_LR true
        self.autosave_n_files = autosave_n_files # number of different save-files (cylces)
        self.autosave_current=0 #...file
        self.autosave_last_saved = time.clock()
        self.autosave_last_printed = time.clock()
        self.LR_start=LR_start
        self.LR_end=LR_end
        self.LR_totalDiff = LR_end-LR_start
        self.CNN=cnn
        self.exponential_interpolation=exponential_interpolation
        print "Autosaver:: saving interval set to",self.autosave_frequency,"minutes"#. Total training time is:",self.training_termination_time/60.0,"minutes (",self.training_termination_time/60.0**2,"h)"
        #print "AutosaveControl:: Controls now the momentum! scanning from 0.5 to 0.95 (reaches 0.9 after 80% of the time passed)"

#        print "AUTO-LR-Control:: initial LR =",LR_start,"final LR will be:",LR_end,(". Linear interpolation." if not exponential_interpolation else ". Exponential interpolation (decay).")
        #######################################################################

    def time_passed(self):
        return (time.clock() - self.training_start_time)

    def tick(self, iter_count, additional_save_string = "", force_save = False, update_LR = True):
        """ returns True if training should END"""
        if (time.clock() - self.autosave_last_saved)/60 >= self.autosave_frequency or force_save:
            self.autosave_last_saved = time.clock()
            self.CNN.SaveParameters((self.save_name)+'_'+str(self.autosave_current)+"__"+str(additional_save_string)+".save")
            self.autosave_current = (self.autosave_current+1) % self.autosave_n_files

        if (time.clock() - self.training_start_time >= self.training_termination_time):
            return True

        if (time.clock() -self.autosave_last_printed) > self.status_printing_interval:
            if update_LR:
                if not self.exponential_interpolation:
                    new_LR = self.LR_start +self.ratio_done()*self.LR_totalDiff
                else:
                    new_LR = self.LR_start*((self.LR_end/self.LR_start)**self.ratio_done())
                self.CNN.set_SGD_LR(new_LR)
            #self.CNN.set_SGD_Momentum(0.5 + n.sqrt(self.ratio_done())*0.45)
            self.autosave_last_printed = time.clock()
            tmin = ((time.clock()-self.training_start_time)/60.)
#            print self.ratio_done()*100.0,' % done'
            print "Training time so far =",int(tmin/60),'h',int(tmin%60),"min. Speed:",(iter_count/(time.clock()-self.training_start_time)),"batches/s"

        return False
        #######################################################################

    def ratio_done(self):#0 to 1
        return (time.clock() - self.training_start_time)/self.training_termination_time

    def set_lr(self,lr):
        self.LR_start = np.float32(lr)
        self.LR_totalDiff =  self.LR_start/10.0 - self.LR_start
        #######################################################################
###############################################################################

#----------------------------------------------------------------------














