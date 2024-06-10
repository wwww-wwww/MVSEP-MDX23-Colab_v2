#@markdown #Separation
from pathlib import Path

#@markdown ---
#@markdown #### separation config:
input = ""  #@param {type:"string"}
output_folder = 'out'  #@param {type:"string"}

import os
import inference

os.makedirs(output_folder, exist_ok=True)

output_format = 'FLAC'  #@param ["PCM_16", "FLOAT", "FLAC"]
Separation_mode = 'Vocals/Instrumental'  #@param ["Vocals/Instrumental", "4-STEMS"]
input_gain = 0  #@param [0, -3, -6] {type:"raw"}
restore_gain_after_separation = True  #@param {type:"boolean"}
filter_vocals_below_50hz = False  #@param {type:"boolean"}
#@markdown ___
##@markdown

#@markdown  ### Model config:

#@markdown  *Set BigShifts=1 to disable that feature*
BigShifts = 3  #@param {type:"slider", min:1, max:41, step:1}
#@markdown ---
BSRoformer_model = 'ep_317_1297'  #@param ["ep_317_1297", "ep_368_1296"]
weight_BSRoformer = 10  #@param {type:"slider", min:0, max:10, step:1}
##@markdown ---
weight_InstVoc = 4  #@param {type:"slider", min:0, max:10, step:1}
#@markdown ---
use_VitLarge = True  #@param {type:"boolean"}
weight_VitLarge = 1  #@param {type:"slider", min:0, max:10, step:1}
#@markdown ---
use_InstHQ4 = False  #@param {type:"boolean"}
weight_InstHQ4 = 2  #@param {type:"slider", min:0, max:10, step:1}
overlap_InstHQ4 = 0.1  #@param {type:"slider", min:0, max:0.95, step:0.05}
#@markdown ---
use_VOCFT = False  #@param {type:"boolean"}
weight_VOCFT = 2  #@param {type:"slider", min:0, max:10, step:1}
overlap_VOCFT = 0.1  #@param {type:"slider", min:0, max:0.95, step:0.05}
#@markdown ---
#@markdown  *Demucs is only used in 4-STEMS mode.*
overlap_demucs = 0.6  #@param {type:"slider", min:0, max:0.95, step:0.05}

use_InstVoc_ = '--use_InstVoc'  #forced use
use_BSRoformer_ = '--use_BSRoformer'  #forced use
use_VOCFT_ = '--use_VOCFT' if use_VOCFT is True else ''
use_VitLarge_ = '--use_VitLarge' if use_VitLarge is True else ''
use_InstHQ4_ = '--use_InstHQ4' if use_InstHQ4 is True else ''
restore_gain = '--restore_gain' if restore_gain_after_separation is True else ''
vocals_only = '--vocals_only' if Separation_mode == 'Vocals/Instrumental' else ''
filter_vocals = '--filter_vocals' if filter_vocals_below_50hz is True else ''

if Path(input).is_file():
  file_path = input
  Path(output_folder).mkdir(parents=True, exist_ok=True)
  inference.predict_with_model({
      "input_audio": [file_path],
      "large_gpu": False,
      "BSRoformer_model": BSRoformer_model,
      "weight_BSRoformer": weight_BSRoformer,
      "weight_InstVoc": weight_InstVoc,
      "weight_InstHQ4": weight_InstHQ4,
      "weight_VOCFT": weight_VOCFT,
      "weight_VitLarge": weight_VitLarge,
      "overlap_demucs": overlap_demucs,
      "overlap_VOCFT": overlap_VOCFT,
      "overlap_InstHQ4": overlap_InstHQ4,
      "output_format": output_format,
      "BigShifts": BigShifts,
      "output_folder": output_folder,
      "input_gain": input_gain,
      "use_InstVoc": True,
      "use_BSRoformer": True,
      "use_VOCFT": use_VOCFT,
      "use_VitLarge": use_VitLarge,
      "use_InstHQ4": use_InstHQ4,
      "restore_gain": restore_gain_after_separation,
      "vocals_only": Separation_mode == "Vocals/Instrumental",
      "filter_vocals": filter_vocals_below_50hz,
  })
