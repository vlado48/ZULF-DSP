import numpy as np
from ZulfCore import ZulfCore

class Spectrum(ZulfCore):
    """
    Postprocesses time signal data and holds the data after each operation.
    
    Spectrum class conducts postprocessing operations on time signal data as
    per user defined settings. Class inherits most of its methods from
    :class:`ZulfCore`, mainly due to readability. Each postprocessing stage is
    conducted by instantiating :object:`~.PostprocessingStep` that will process
    and then hold data as attributes.
        :class:`Spectrum` will hold copy of data after *each* postprocessing
    step in :list:`~.postprocessed`.
    """
    
    
    
    def __init__(self, **kwargs):
        """
        Instantiates attribudes of base class :class:`ZulfCore` to attain
        :list:`ZulfCore.operations` that holds postprocessing methods
        wrapped in :dataclass:`ZulfCore.Operation`.
        Subsequently updates user settings that have been passed as constructor
        kwargs and then calls :method:`~.__postprocess`.

        Parameters (Operation disabling/enabling)
        ----------------------------------------
        cut : bool, default = True
            Enable or disable removal of front datapoints of time signal.
        debase : bool, default = True
            Enable or disable removal signal background.
        filter : bool, default = True
            Enable or disable signal filtering.
        predict : bool, default = True
            Enable or disable backward linear prediction.
        zerofill : bool, default = True
            Enable or disable zero-padding.
        apodize : bool, default = True
            Enable or disable apodization of time signal.
        phase_correct : bool, default = True
            Enable or disable phase correction of spectrum.       
        
        Parameters (Operation settings)
        -------------------------------
        data_src : str or iterable, default = "manual"
            Determines source of time-signal data:
            - "manual" prompts file selection dialog.
            - "systempath\filename" gives path to .dat file on local drive.
            - iterable must contain two array_like object holding timestamps
                and signal values.
        ms_to_cut : int, default = 0
            Number of miliseconds of time signal to be removed from the front
            of time signal.
        fit_order : int, default = 20
            Degree of polynomial fit to be substracted from time signal in 
            debasing procedure.
        filter_freq : iterable of floats
            Lists frequencies to be filtered in filtering stage.
            Default = (50, 100, 150, 200, 250, 300)
        prediction_fill : int or "auto", default = "auto"
            Number of datapoints to be created via backward prediction.
        prediction_order : int or "auto", default = "auto"
            Order of backward prediction filter.
        prediction_scan : int or "all", default = "all"
            N of points from the beggining of time signal used to model the
            prediction filter. It is recommended to use own value.
        decay_c : float, default = -0.25
            Coeffecient of decaying exponenctial c as in x*e^c used for
            apodization of time signal.
        phase_shift : float or "manual", default = 0
            Phase shift used to correct the spectrum's phase in degrees.
            "manual" option prompts an interactive plot.
        """
    
        ZulfCore.__init__(self)
        self.kwargs = kwargs
        
        self.update_settings(kwargs)
        self.postprocessed = []        
        self.__postprocess(0)

    def __postprocess(self, start):
        """
        Iteratively calls all enabled operations by using
        :method:`ZulfCore.Operation.execute` via instantiating
        :class:`~.PostprocessingStep`. Each `PostprocessingStep` instance holds
        data from given stage and is stored in a :list:`~.postprocesed` for
        later access.
        
        Parameters
        ----------
        start : int
            Index of starting operation. Certain methods may want to call
            postprocess only from certain stage.
            
        TODO: implement :method:`~.iterate`
        TODO: change :param:`start` from index to operation name as index is 
              arbitrary.
 
        """
        
        # Create list of only enabled operations
        mask = [operation.enabled for operation in self.operations]
        enabled_operations = np.array(self.operations)[mask]
        
        # Ensure that old data are not kept for operations to be executed this
        # postprocess() call.
        if len(self.postprocessed) != 0:
            self.postprocessed = self.postprocessed[:start]

        class PostprocessingStep:
            """
            Class executes operation and keeps results as atributes.
            
            Upon instantiation class pulls most recent data and executes
            indicated :param:`operation`. Postprocessed data are kept as 
            attribudes :atr:`PostprocessingStep.x / y / fft`. Instance of this
            class has to keep referece to :object:`Spectrum` in 
            :param:`outer_instance` to be adress it's owner instance.
            
            Attributes
            ----------
            outer_instance : object
                reference to owner instance of :object:`Spectrum`.
            operation : object
                reference to the operation that was conducted by instantiation
                of this `PostprocessingStep` object.
            
            Note
            ----
            Spectrum owns PostprocessingStep as an attribute, it is
            not is's derived class.
            """
            
            def __init__(self, outer_instance, operation):
                self.outer_instance = outer_instance
                self.operation = operation
                
                x, y, fft = get_last_values(self, operation)
                x, y, fft = operation.execute(x, y, fft)
                self.x = x
                self.y = y
                self.fft = fft

        def get_last_values(self, current_op):
            """Returns data from most recet postprocessing operation"""
 
            if current_op != self.outer_instance.open_op:
                current_i = list(enabled_operations).index(current_op)
                previous_i = current_i - 1
                previous_step = self.outer_instance.postprocessed[previous_i]
                return previous_step.x, previous_step.y, previous_step.fft
            return None, None, None            


        for operation in enabled_operations:
            print(operation.name)
            step = PostprocessingStep(self, operation)
            self.postprocessed.append(step)
            
    