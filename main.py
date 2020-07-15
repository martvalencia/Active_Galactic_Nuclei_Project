import aplpy
import astroscrappy
import ccdproc
import glob
import numpy as np
import os
import sep
from reproject import reproject_interp
import ois
from astropy import units as u
from astropy.io import fits
from photutils import make_source_mask

dark_flat_dir = "/home/linkmaster/Documents/CTMO/Projects/Pipeline/AGN_Project/Darks/Flats"


dark_obj_dir = "/home/linkmaster/Documents/CTMO/Projects/Pipeline/AGN_Project/Darks/Object"


flat_dir = "/home/linkmaster/Documents/CTMO/Projects/Pipeline/AGN_Project/Flats"

object_dir = "/home/linkmaster/Documents/CTMO/Projects/Pipeline/AGN_Project/Object"


output_dir ="/home/linkmaster/Documents/CTMO/Projects/Pipeline/AGN_Project/Output"
save_cal = True

###########################################################################################
# Reading dark flat frames and median comining them
dark_flat_list = []

# Change to dark directory
os.chdir(dark_flat_dir)

# Append frames to dark list
for frame in glob.glob("*.fit"):
    dark_flat_list.append(frame)
dark_flat_list.sort()

# Median combine dark frames
master_dark_flat = ccdproc.combine(dark_flat_list, 
                                   method="median", 
                                   unit="adu", 
                                   mem_limit=6e9)
master_dark_flat_data = np.asarray(master_dark_flat)

# Display FITSFigure
fig = aplpy.FITSFigure(master_dark_flat_data)
fig.show_grayscale()
fig.add_colorbar()

###########################################################################################


dark_obj_list = []

# Change to dark directory
os.chdir(dark_obj_dir)

# Append frames to dark list
for frame in glob.glob("*.fit"):
    dark_obj_list.append(frame)
dark_obj_list.sort()

# Median combine dark frames
master_dark_obj = ccdproc.combine(dark_obj_list, 
                                  method="median", 
                                  unit="adu", 
                                  mem_limit=6e9)
master_dark_obj_data = np.asarray(master_dark_obj)

# Display FITSFigure
fig = aplpy.FITSFigure(master_dark_obj_data)
fig.show_grayscale()
fig.add_colorbar()

###########################################################################################

flat_list = []

# Change to flat directory
os.chdir(flat_dir)

# Append frames to flat list
for frame in glob.glob("*.fit"):
    flat_list.append(frame)
flat_list.sort()

# Median combine flat frames
combined_flat = ccdproc.combine(flat_list, 
                                method="median", 
                                unit="adu", 
                                mem_limit=6e9)

# Subtract master dark from combined flat
master_flat = ccdproc.subtract_dark(combined_flat, 
                                    master_dark_flat, 
                                    data_exposure=combined_flat.header["exposure"]*u.second, 
                                    dark_exposure=master_dark_flat.header["exposure"]*u.second, 
                                    scale=True)

# Convert dark-reduced flat to data array
master_flat_data = np.asarray(master_flat)

# Normalizing flatfield array by its mean
array_mean = np.mean(master_flat_data)
flatfield = master_flat_data / array_mean

# Display FITSFigure
fig = aplpy.FITSFigure(flatfield)
fig.show_grayscale()
fig.add_colorbar()

###########################################################################################

obj_list = []

# Change to object directory
os.chdir(object_dir)

# Append frames to object list
for frame in glob.glob("*.fit"):
    obj_list.append(frame)
obj_list.sort()

for item in obj_list:
    
    # Read object frame
    obj_frame = fits.open(item)
    
    # Read object data and header
    obj_data = obj_frame[0].data
    obj_header = obj_frame[0].header
    
    # Subtract master dark from object
    obj_min_dark = obj_data - master_dark_obj

    # Divide object by flatfield frame
    reduced_data = obj_min_dark / flatfield
    
    # Mask sources
    mask = make_source_mask(reduced_data, 
                           nsigma=2.0,
                           npixels=5,
                           dilate_size=31)
    
    # Subtract background
    background = sep.Background(reduced_data, mask=mask)
    reduced_data -= background

    # Save reduced frame
    if save_cal == True:
        hdu = fits.PrimaryHDU(reduced_data, header=obj_header)
        hdu.writeto(output_dir + "/reduced-" + item, overwrite=True)
        
# Display last FITSFigure
fig = aplpy.FITSFigure(reduced_data)
fig.show_grayscale()
fig.add_colorbar()
fig.add_label(650, 100, "reduced-" + item, color="white")

###########################################################################################
#stacking images
stack_list =[]

os.chdir(object_dir)

# Append frames to object list
for frame in glob.glob("*.fit"):
    stack_list.append(frame)
stack_list.sort()

stack_header = stack_list[0].header
stack_data = stack_list[0].data

stack_to_list=[]
for image in stack_list:
    stack_to_list.append(fits.getdata(image))

final_image = np.sum(stack_to_list, axis=0)

if save_cal == True:
    hdu = fits.PrimaryHDU(final_image, header = stack_header)
    hdu.writeto(output_dir + "/stacked_image" + final_image, overwrite = True)
    

###########################################################################################

#Difference Imaging

ref_fits = fits.open("/home/linkmaster/Documents/CTMO/Projects/Pipeline/AGN_Project/Reference/dss_search")
targ_fits = fits.open("final_stacked_image.fit") # stack image FITS image

# Read FITS data
ref_data = fits.getdata("/home/linkmaster/Documents/CTMO/Projects/Pipeline/AGN_Project/Reference/dss_search")
targ_data = fits.getdata("final_stacked_image.fit")

# Read FITS headers
ref_hdr = ref_fits[0].header
targ_hdr = targ_fits[0].header

# Alignment
aligned_array, footprint = reproject_interp(targ_fits, ref_hdr) #aligned array is the new fits file aligned
print(type(aligned_array))
aligned_hdu = fits.PrimaryHDU(aligned_array, header=ref_hdr)
aligned_hdu.writeto("aligned_to.fit", overwrite=True)

# Subtraction
diff_data, optimal_image, kernel, background = ois.optimal_system(aligned_array, ref_data, kernelshape = (11,11), method = "Bramich") 
hdu_diff = fits.PrimaryHDU(diff_data, header=ref_hdr)
hdu_diff.writeto("sub.fit", overwrite=True)
