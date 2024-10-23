import sys
sys.path.append('utility_box/')

import ocv
from cpath import WSI, CPDataset
from shapely_utils import (
    make_valid, 
    get_background, 
    get_geoms_from_mask, 
    get_geom_slicing_bounds, 
    slice_geom, 
    get_geom_coordinates,
    sample_from_geom, 
    get_box,
    remove_duplicates_valid,
    flatten_geoms
)
from shapely.ops import unary_union
from shapely_utils import GeometryCollection, MultiPolygon, Polygon
import load

import pandas as pd
import random
from tqdm.auto import tqdm
import cv2
import numpy as np
from pathlib import Path

import torch

class CamelyonDataset(CPDataset):
    def __init__(
        self,
        data_distribution:pd.DataFrame,
        patch_size:int,
        overlap:int,
        target_mpp:float,
        training_mode:str,
        slice_tumor:bool=False,
        total_patches:int=100,
        sample_without_replacement:bool=True,
        n_sample_rows:int=2,
        augmentations=None
    ):
        super(CamelyonDataset, self).__init__(data_distribution)
        
        self.limit_choices=[(20,180), (20,200)]#,(20,220)]
        self.training_mode=training_mode
        self.target_mpp=target_mpp
        self.patch_size=patch_size
        self.overlap=overlap
        self.flag=0
        self.sample_without_replacement=sample_without_replacement
        self.geom_dicts=[]
        self.n_sample_rows=n_sample_rows
        self.slice_tumor=slice_tumor
        self.total_patches=total_patches
        self.positive_probability=len(self.positive_cases)/(len(self.negative_cases)+len(self.positive_cases))
        self.n_patches_list=self.sample_patches()
        self.augmentations=augmentations
        self.sampled_wsis={}

        if self.training_mode=='pure_positive':
            sample_rows=self.sample_positive_rows(self.n_sample_rows, replace=True)
            pbar=tqdm(total=len(sample_rows), desc='Sampling Rows')
            for idx, sample_row in enumerate(sample_rows):
                self.process_row_sample(
                    sample_row,
                    slice_tumor=self.slice_tumor,
                    sample_tissue_patches=False,
                    sample_tumor_patches=(not self.slice_tumor),
                    n_sample_patches=self.n_patches_list[idx],
                )
                pbar.update()
                
        if self.training_mode=='pure_negative':
            sample_rows=self.sample_negative_rows(self.n_sample_rows, replace=True)
            pbar=tqdm(total=len(sample_rows), desc='Sampling Rows')
            for idx, sample_row in enumerate(sample_rows):
                self.process_row_sample(
                    sample_row,
                    slice_tumor=False,
                    sample_tissue_patches=True,
                    sample_tumor_patches=False,
                    n_sample_patches=self.n_patches_list[idx],
                )
                pbar.update()
        
        if self.training_mode=='mixed':
            pbar=tqdm(total=n_sample_rows, desc='Sampling Rows')
            for idx in range(n_sample_rows):
                if 0 in random.choices([0, 1], weights=[1-self.positive_probability, self.positive_probability]):
                    sample_row=self.sample_negative_rows(1)[0]
                    self.process_row_sample(
                        sample_row,
                        slice_tumor=False,
                        sample_tissue_patches=True,
                        sample_tumor_patches=False,
                        n_sample_patches=self.n_patches_list[idx],
                    )
                else:
                    sample_row=self.sample_positive_rows(1)[0]
                    self.process_row_sample(
                        sample_row,
                        slice_tumor=self.slice_tumor,
                        sample_tissue_patches=False,
                        sample_tumor_patches=(not self.slice_tumor),
                        n_sample_patches=self.n_patches_list[idx],
                    )
                pbar.update()
                    
    def sample_patches(self):
        patches_per_row=self.total_patches // self.n_sample_rows
        remainder=self.total_patches % self.n_sample_rows
        n_patches_list=[patches_per_row] * (self.n_sample_rows-1)
        n_patches_list.append(patches_per_row+remainder)
        
        return n_patches_list

    def process_row_sample(
        self,
        sample_row,
        slice_tumor=False,
        sample_tissue_patches=False,
        sample_tumor_patches=False,
        n_sample_patches=None
        ):

        self.sampled_wsis[sample_row['wsi_name']]=[]
        wsi_path=Path(f"{sample_row['wsi_folder']}/{sample_row['wsi_name']}")
        if wsi_path.exists():
            wsi=WSI(wsi_path, mpp=sample_row['mpp'])
        else:
            raise FileNotFoundError(f"The file does not exist: {wsi_path}")
            
        scale,rescale=wsi.scale_mpp(self.target_mpp)
        limits=random.choice(self.limit_choices)
        tissue_mask, tissue_mpp=wsi.get_tissuemask_fast(limits)
        tissue_scale,tissue_rescale=wsi.scale_mpp(tissue_mpp)
        tissue_geoms=get_geoms_from_mask(tissue_mask, tissue_rescale)
        tissue_mgeom=MultiPolygon(tissue_geoms).buffer(0)
        background_geom=get_background(tissue_mgeom)

        tumor_geoms=[]
        hole_geoms=[]
        
        if sample_row['group'] !='negative':
            tumor_geom_dicts=load.load_pickle(f"{sample_row['ann_folder']}/{sample_row['ann_name']}")
            
            for tumor_geom_dict in tumor_geom_dicts:
                if tumor_geom_dict['group']=='Tumor':
                    tumor_geoms.extend(flatten_geoms(tumor_geom_dict['geom']))
                else:
                    hole_geoms.extend(flatten_geoms(tumor_geom_dict['geom']))

            tumor_mgeom=MultiPolygon(tumor_geoms).buffer(0)
            
            cleaned_tumor_geoms=[]
            for geom in tumor_geoms:
                with_hole=False
                for hole in hole_geoms:
                    if geom.intersects(hole):
                        with_hole=True
                        cleaned_tumor_geoms.append(geom.difference(hole))
                if not with_hole:
                    cleaned_tumor_geoms.append(geom)
            
        
            background_geom=background_geom.difference(tumor_mgeom)      
            tissue_mgeom=tissue_mgeom.difference(tumor_mgeom)
            if len(hole_geoms)>0:
                tissue_mgeom=tissue_mgeom.difference(MultiPolygon(hole_geoms).buffer(0))
            
        else:
            cleaned_tumor_geoms=None
            tumor_mgeom=None
        
        geom_dict={}
        geom_dict['wsi_name']=sample_row['wsi_name']
        geom_dict['background_geom']=background_geom
        geom_dict['cleaned_tumor_geoms']=cleaned_tumor_geoms
        geom_dict['tissue_mgeom']=tissue_mgeom
        geom_dict['tumor_mgeom']=tumor_mgeom
        geom_dict['rescaled_overlap']=int(self.overlap*rescale)
        geom_dict['rescaled_patch_size']=int(self.patch_size*rescale)
        geom_dict['wsi']=wsi
        geom_dict['tumor_geoms']=tumor_geoms
        geom_dict['hole_geoms']=hole_geoms
        geom_dict['coords_list']=[]

        if slice_tumor:
            sampled_coords=slice_geom(
                geom=geom_dict['tumor_mgeom'],
                geom_limit=geom_dict['background_geom'],
                patch_size=geom_dict['rescaled_patch_size'],
                overlap=geom_dict['rescaled_overlap']
            )
            geom_dict['coords_list']=geom_dict['coords_list']+sampled_coords
            self.flag+=len(sampled_coords)
            
        if sample_tissue_patches and sample_tumor_patches:
            
            sampled_coords=sample_from_geom(
                geom=geom_dict['tissue_mgeom'],
                geom_limit=geom_dict['background_geom'],
                patch_size=geom_dict['rescaled_patch_size'],
                overlap=geom_dict['rescaled_overlap'],
                n_samples=n_sample_patches//2
            )
            
            geom_dict['coords_list']=geom_dict['coords_list']+sampled_coords
            self.flag+=len(sampled_coords)
            
            sampled_coords=sample_from_geom(
                geom=geom_dict['tumor_mgeom'],
                geom_limit=geom_dict['background_geom'],
                patch_size=geom_dict['rescaled_patch_size'],
                overlap=geom_dict['rescaled_overlap'],
                n_samples=(n_sample_patches//2)+(n_sample_patches%2)
            )
            geom_dict['coords_list']=geom_dict['coords_list']+sampled_coords
            self.flag+=len(sampled_coords)
        
        if sample_tissue_patches:
            sampled_coords=sample_from_geom(
                geom=geom_dict['tissue_mgeom'],
                geom_limit=geom_dict['background_geom'],
                patch_size=geom_dict['rescaled_patch_size'],
                overlap=geom_dict['rescaled_overlap'],
                n_samples=n_sample_patches
            )
            geom_dict['coords_list']=geom_dict['coords_list']+sampled_coords
            self.flag+=len(sampled_coords)

        if sample_tumor_patches:
            sampled_coords=sample_from_geom(
                geom=geom_dict['tumor_mgeom'],
                geom_limit=geom_dict['background_geom'],
                patch_size=geom_dict['rescaled_patch_size'],
                overlap=geom_dict['rescaled_overlap'],
                n_samples=n_sample_patches
            )
            geom_dict['coords_list']=geom_dict['coords_list']+sampled_coords
            self.flag+=len(sampled_coords)
            
        geom_dict['sampling_legend']=np.arange(0, len(geom_dict['coords_list'])).tolist()
        self.geom_dicts.append(geom_dict)
    
    def __len__(self):
        return self.flag

    def __getitem__(self, idx):
        
        random_geom_dict=random.choice(self.geom_dicts)
        random_idx=random.choice(random_geom_dict['sampling_legend'])
        
        x,y=random_geom_dict['coords_list'][random_idx]
        self.sampled_wsis[random_geom_dict['wsi_name']].append((x,y))
        
        if self.sample_without_replacement:
            random_geom_dict['sampling_legend'].remove(random_idx)
        
        if random_geom_dict['cleaned_tumor_geoms'] is None:
            
            mask=np.zeros((random_geom_dict['rescaled_patch_size'],random_geom_dict['rescaled_patch_size']), dtype=np.uint8)
            patch=random_geom_dict['wsi'].get_patch(
                x,
                y,
                random_geom_dict['rescaled_patch_size'],
                random_geom_dict['rescaled_patch_size']
            )
            
        else:
            try:
                sampled_box=get_box(x,y,random_geom_dict['rescaled_patch_size'],random_geom_dict['rescaled_patch_size'])
                mask=np.zeros((random_geom_dict['rescaled_patch_size'],random_geom_dict['rescaled_patch_size']), dtype=np.uint8)
                    
                for geom in random_geom_dict['cleaned_tumor_geoms']:
                    if geom.intersects(sampled_box):
                        tumor_contours, tumor_holes=get_geom_coordinates(geom.intersection(sampled_box))
                        for contour in tumor_contours:
                            shifted_coords = np.array([(x_-int(sampled_box.bounds[0]), y_-int(sampled_box.bounds[1])) for x_, y_ in np.array(contour)], dtype=np.int32)
                            cv2contour=shifted_coords.reshape((-1, 1, 2))
                            cv2.fillPoly(mask, [cv2contour], color=1)
                                
                        for contour in tumor_holes:
                            shifted_coords = np.array([(x_-int(sampled_box.bounds[0]), y_-int(sampled_box.bounds[1])) for x_, y_ in np.array(contour)], dtype=np.int32)
                            cv2contour=shifted_coords.reshape((-1, 1, 2))
                            cv2.fillPoly(mask, [cv2contour], color=0)
                
                patch=random_geom_dict['wsi'].get_patch(
                    int(sampled_box.bounds[0]),
                    int(sampled_box.bounds[1]),
                    random_geom_dict['rescaled_patch_size'],
                    random_geom_dict['rescaled_patch_size']
                    )
            except Exception as e:
                mask=np.zeros((random_geom_dict['rescaled_patch_size'],random_geom_dict['rescaled_patch_size']), dtype=np.uint8)
                patch=np.zeros((random_geom_dict['rescaled_patch_size'],random_geom_dict['rescaled_patch_size'], 3), dtype=np.uint8)
                print(f"dataloading_error:{e}")
        
        patch=cv2.resize(patch, (self.patch_size, self.patch_size))
        mask=cv2.resize(mask, (self.patch_size, self.patch_size))
        
        if self.augmentations:
            sample = self.augmentations(image=patch, mask=mask)
            patch, mask = sample["image"], sample["mask"]
            
        torch_patch=torch.as_tensor(patch.transpose((2, 0, 1)).copy()).float().contiguous()
        torch_mask=torch.as_tensor(mask.copy()).long().contiguous()
        
        return torch_patch/255, torch_mask
        

class InferenceDataset(CPDataset):
    
    def __init__(
        self,
        data_distribution, 
        patch_size, 
        overlap,
        target_mpp,
        slice_startegy='whole_slide',
        inference_mode='random_sample' #
    ):
        super(InferenceDataset, self).__init__(data_distribution)

        if inference_mode not in ['positive_sample', 'negative_sample', 'random_sample']:
            raise ValueError

        self.patch_size=patch_size
        self.overlap=overlap
        self.target_mpp=target_mpp
        self.inference_coords=[]
        self.slice_startegy=slice_startegy

        if inference_mode=='random_sample':
            samples=self.sample_mixed_rows(1)
        if inference_mode=='positive_sample':
            samples=self.sample_positive_rows(1)
        if inference_mode=='negative_sample':
            samples=self.sample_negative_rows(1)
        if inference_mode=='custom_sample':
            pass
            
        for sample in samples:
            self.process_row_sample(sample)

    def process_row_sample(
        self, 
        sample=None
    ):
        if self.slice_startegy=='whole_slide':
            if sample is not None:
                self.wsi_path=Path(f"{sample['wsi_folder']}/{sample['wsi_name']}")
                wsi=WSI(self.wsi_path, sample['mpp'])
                self.wsi=wsi
                self.slice_whole_wsi()

        
    def slice_whole_wsi(self):
    
        scale,rescale=self.wsi.scale_mpp(self.target_mpp)

        self.scale=scale
        self.rescale=rescale
        self.rescaled_overlap=int(self.overlap*rescale)
        self.rescaled_patch_size=int(self.patch_size*rescale)
        
        step_size=int(self.rescaled_patch_size-self.rescaled_overlap)
        
        x_lim, y_lim = self.wsi.dims
        
        for x in range(0, x_lim, step_size):
            if x+self.rescaled_patch_size>x_lim:
                x=x_lim-self.rescaled_patch_size
            for y in range(0, y_lim, step_size):
                if y+self.rescaled_patch_size>y_lim:
                    y=y_lim-self.rescaled_patch_size
                self.inference_coords.append((x, y))
                
    def __len__(self):
        return len(self.inference_coords)

    def __getitem__(self, idx):
        x, y = self.inference_coords[idx]
        
        patch=self.wsi.get_patch(
            x,
            y,
            self.rescaled_patch_size,
            self.rescaled_patch_size
            )
        
        patch=cv2.resize(patch, (self.patch_size, self.patch_size))
        torch_patch=torch.as_tensor(patch.transpose((2, 0, 1)).copy()).float().contiguous()
        
        return torch_patch/255


