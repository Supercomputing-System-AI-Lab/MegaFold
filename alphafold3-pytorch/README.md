# AF3 training code

## Install required dependencies

```
# create virtual environment under python==3.13 and activate 
conda create -n venv python==3.13.0
conda activate venv 

# install torch==2.7.0+cu11.8
pip install torch==2.7.0  --index-url https://download.pytorch.org/whl/cu118

# install other packages
pip install -r requirements.txt
```

---
## Prepare experiment dataset

First, download a sample dataset from the Protein Data Bank (PDB). 

```
wget "https://mailmissouri-my.sharepoint.com/:u:/g/personal/acmwhb_umsystem_edu/ESbEXPguyO9Moh3E_J1zkWQBXZ6JxE5bsoKrZXOVwtu1Ow?download=1" -O data/pdb_data/val_mmcifs.tar.gz
tar -xzf data/pdb_data/val_mmcifs.tar.gz -C data/pdb_data
rm data/pdb_data/val_mmcifs.tar.gz
```

Then, install required MSAs and templates data.

```
# install msa_dir
wget "https://mailmissouri-my.sharepoint.com/:u:/g/personal/acmwhb_umsystem_edu/EbXU1bnlRZxIqUXbAprgHycB3F4GWLy-m-qxvODfJsvFvA?download=1" -O pdb_val_msas
tar -xvzf pdb_val_msas
cp -r scratch/references/af3/pdb_data/* data/pdb_data/
rm pdb_val_msas
rm -r scratch

# install templates_dir
wget "https://umass-my.sharepoint.com/:u:/g/personal/hvla_umass_edu/EUalS7Hq3KBOlGdF2bVVwFABYU_ZidT2nEEi0PwqxaZ_Fw?download=1" -O templates_dir
tar -xvzf templates_dir 
cp -r scratch/references/af3/pdb_data/* data/pdb_data/
rm templates_dir
rm -r scratch
```

Then, install PDB's Chemical Component Dictionary (CCD) and miscellaneous metadata. 

```
# install CCD data
wget -P ./data/ccd_data/ https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz
wget -P ./data/ccd_data/ https://files.wwpdb.org/pub/pdb/data/component-models/complete/chem_comp_model.cif.gz
gunzip data/ccd_data/components.cif.gz
gunzip data/ccd_data/chem_comp_model.cif.gz

# install misc_data
wget "https://mailmissouri-my.sharepoint.com/:u:/g/personal/acmwhb_umsystem_edu/ESb9kUT_ASBEsYRN0KQmqt4BLzJhFunQU86E-GxWGxtGiA?download=1" -O misc_data
tar -xzf misc_data -C data/pdb_data
rm misc_data
```

Now, download the cache of deterministic features, used in Ahead-of-Time Cache-based Data-Loading Optimization.

```
# install msa_cache_dir
wget "https://mailmissouri-my.sharepoint.com/:u:/g/personal/acmwhb_umsystem_edu/Ect3VyxyqnZPm-4I6EpzB64B2M6tGctY5OMjIkatr6kYHQ?download=1" -O msa_cache
tar -xvzf msa_cache --wildcards 'caches/pdb_data/cache/msa/val_msas*'
rm msa_cache

# install input_cache_dir 
wget "https://mailmissouri-my.sharepoint.com/:u:/g/personal/acmwhb_umsystem_edu/EXQnFYxhepNNku_Df45B1gEBPlhzIH_RtnhUEae4b74SKQ?download=1" -O input_cache
tar -xvzf input_cache 
rm input_cache
```

---
## Run code

- Baseline: 

``` 
AF3_OPTIMIZATIONS_MODE="baseline" python3 train.py --config configs/baseline.yaml --trainer_name initial_training
```


- MegaFold: 

``` 
AF3_OPTIMIZATIONS_MODE="megafold" python3 train.py --config configs/megafold.yaml --trainer_name initial_training
```

*NOTE*: Scripts to submit batch jobs are available in `bash-scripts`.


---
## (optional) Full dataset & cache:

If you are interested in running large-scale AlphaFold3 training, the full dataset and its cache are provided below:  

```
# download `omniflow_caches.tar.gz.part_{aa,ab}` and `omniflow_data.tar.gz` from SharePoint
wget "https://mailmissouri-my.sharepoint.com/:u:/g/personal/acmwhb_umsystem_edu/Ect3VyxyqnZPm-4I6EpzB64B2M6tGctY5OMjIkatr6kYHQ?download=1"
wget "https://mailmissouri-my.sharepoint.com/:u:/g/personal/acmwhb_umsystem_edu/ERiOg_fC_6BFnr9oKilzeeUBz8O_a2tI0i-TlksYAf8E5g?download=1"
wget "https://mailmissouri-my.sharepoint.com/:u:/g/personal/acmwhb_umsystem_edu/EYQ9oFu5KmFLryp8F1m79BAB2zoUFtLIU-Bx2OWmmKAdtA?download=1"

# then reassemble, extract, and clean up the downloaded archives
cat omniflow_caches.tar.gz.part_* > omniflow_caches.tar.gz
tar -xzf omniflow_caches.tar.gz && rm omniflow_caches.tar.gz
tar -xzf omniflow_data.tar.gz && rm omniflow_data.tar.gz
```