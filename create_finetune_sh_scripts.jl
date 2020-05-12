for i in 1:2
    for weight=0.0:0.01:0.10
        open("results/finetune_erw$(i)_train_$(weight).sh", "w") do file
            write(file, """#!/bin/sh
                  #SBATCH --gres=gpu:0
                  #SBATCH --partition=Luna,Gobi
                  #SBATCH --output=results/erw$(i)_train_$(weight).out
                  #SBATCH --job-name="erw$(i)_train_$(weight)"

                  /home/f/fuchsp/.pyenv/shims/python ../train_finetune.py --data_dir /big/f/fuchsp/data/datasets/train_tr --vocab_dir ../dataset/vocab --model_dir saved_models/clean_ds --erw$(i) True --weight $(weight) --id erw$(i)_train_$(weight) --info "Position-aware attention model"

                  """)
        end
    end
end

