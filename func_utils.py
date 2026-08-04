
from app_utils import (phase_retrieval, background_removal, mu_std_p, training_data_func, training_func,
                       inference_func1, inference_func2)
from app_utils import show_z_psf
import numpy as np
import torch
import os
import time
import scipy.io as sio
import pickle
from skimage import io

def func1(M, NA,  n_immersion, lamda, n_sample, f_4f, ps_camera, ps_BFP, external_mask,
          zstack_file, nfp_text, NFP, zrange, raw_image_folder, snr_roi, max_pv, projection_01,
          num_z_voxel, training_im_size, us_factor, max_num_particles, num_training_images, previous_param_dict, test_idx, threshold,
          state):
    # fetch param_dict
    if 'param_dict' not in state.keys():  # in the case of preprocessing images before characterizing PSF
        state['param_dict'] = dict()
    param_dict = state['param_dict']
    # update param_dict
    param_dict['M'] = M
    param_dict['NA'] = NA
    param_dict['lamda'] = lamda
    param_dict['n_immersion'] = n_immersion
    param_dict['n_sample'] = n_sample
    param_dict['f_4f'] = f_4f
    param_dict['ps_camera'] = ps_camera
    param_dict['ps_BFP'] = ps_BFP
    param_dict['NFP'] = NFP
    param_dict['g_sigma'] = 1.4 #1.4 for microtubule

    # ori's edit
    if "device" not in param_dict:
        raise KeyError("No device was supplied in param_dict. ""Set state['param_dict']['device'] in debugger.py.")
    device = torch.device(param_dict["device"])
    print(f"Device used in func1: {device}")
    param_dict['device'] = device
    #end ori's edit
    #param_dict['device'] = torch.device('cuda:'+str(0) if torch.cuda.is_available() else 'cpu')

    # a dict for phase retrieval
    nfp_text = nfp_text.split(',')
    nfps = np.linspace(float(nfp_text[0]), float(nfp_text[1]), int(nfp_text[2]))
    param_dict["nfps"] = nfps

    pr_dict = dict(
        # zstack_file_path=os.path.join(os.getcwd(), zstack_file),
        zstack_file_path=zstack_file,
        nfps=nfps,
        r_bead=0.2,  # a default value, not critical

        #epoch_num=250//2,  # optimization iterations
        epochs=250//2,  # optimization iterations
        loss_label=1,  # 1: gauss log likelihood, 2: l2
        learning_rate=0.001,  # 0.005?

        # Allowed range for the mask displacement d [um]
        d_bounds_um=(15000.0, 35000.0),
        mask_iris_diameter_mm=5.3,

        # bead-wise robust alignment
        fine_defocus_range_um=0.4,
        fine_defocus_step_um=0.1,
        max_shift_px=15,
    )
    if external_mask == 'None':

        param_dict["mask_iris_diameter_mm"] = float(pr_dict["mask_iris_diameter_mm"])
        phase_mask, g_sigma, ccs = phase_retrieval(param_dict, pr_dict)
        print(f'Phase mask is retrieved. blue sigma: {np.round(g_sigma, decimals=2)}.')
        print(f'PSF modeling accuracy: average cc of {np.round(np.mean(ccs), decimals=4)}.')

    else:
        mask_dict = sio.loadmat(external_mask)
        mask_name = list(mask_dict.keys())[3]
        phase_mask = mask_dict[mask_name]
        g_sigma = 0.6

    param_dict['g_sigma'] = (np.round(0.8*g_sigma, decimals=2), np.round(1.0*g_sigma, decimals=2))
    param_dict['phase_mask'] = phase_mask

    # Ori's edit added on 2/01/2026 for mask displacement optimization
    param_dict['nfps'] = nfps  # save z positions used for stacks

    param_dict['centralBeadCoordinates_pixel'] = param_dict.get('centralBeadCoordinates_pixel', [600, 600])

    # Off-axis stacks are used by the joint phase-retrieval optimization
    # to constrain the field-dependent PSF model.
    param_dict.setdefault('offaxis_zstack_files', [])
    param_dict.setdefault('offaxis_coords_pixel', [])
    param_dict.setdefault('centralBeadCoordinates_pixel', [600, 600])

    param_dict['nfps'] = nfps
    # End  Ori's edit added on 2/01/2026 for mask displacement optimization

    # show z-PSF regarding the NFP
    param_dict['NFP'] = NFP  # now it's not bead
    #zrange = zrange.split(',')
    zrange = (float(zrange[0]), float(zrange[1]))
    param_dict['zrange'] = zrange  # a tuple
    param_dict['baseline'] =  None   # required by imaging model, None for now
    param_dict['read_std'] =  None
    param_dict['bg'] = None
    show_z_psf(param_dict)  # generate PSFs.jpg
    param_dict['non_uniform_noise_flag'] = True
    param_dict['bitdepth'] = 16

    # save param_dict for other blocks
    if 'param_dict' in state.keys():
        state['param_dict'] = {**state['param_dict'], **param_dict}
    else:
        state['param_dict'] = param_dict

    return ("PSF characterization is done. Check "
            "\nphase_retrieval_results.jpg "
            "\nPSFs.jpg")


# background removal
def func2(M, NA,  n_immersion, lamda, n_sample, f_4f, ps_camera, ps_BFP, external_mask,
          zstack_file, nfp_text, NFP, zrange, raw_image_folder, snr_roi, max_pv, projection_01,
          num_z_voxel, training_im_size, us_factor, max_num_particles, num_training_images, previous_param_dict, test_idx, threshold,
          state):  # preprocessing
    # fetch param_dict
    if 'param_dict' not in state.keys():  # in the case of preprocessing images before characterizing PSF
        state['param_dict'] = dict()
    param_dict = state['param_dict']

    im_br_folder = background_removal(raw_image_folder)

    # update param_dict
    param_dict['im_br_folder'] = im_br_folder

    print(f'Images after background removal are in {im_br_folder}.')

    return (f'Background removal is done. Check folder'
            f'\n{im_br_folder}.')


# snr estimation
def func3(M, NA,  n_immersion, lamda, n_sample, f_4f, ps_camera, ps_BFP, external_mask,
          zstack_file, nfp_text, NFP, zrange, raw_image_folder, snr_roi, max_pv, projection_01,
          num_z_voxel, training_im_size, us_factor, max_num_particles, num_training_images, previous_param_dict, test_idx, threshold,
          state):
    # fetch param_dict
    if 'param_dict' not in state.keys():  # in the case of preprocessing images before characterizing PSF
        print('Cannot proceed.')

    else:
        param_dict = state['param_dict']
        if 'phase_mask' not in param_dict.keys():
            print('Cannot proceed. Please characterize PSF] first.')
        else:
            snr_roi = snr_roi.split(',')
            snr_roi = (int(snr_roi[0]), int(snr_roi[1]), int(snr_roi[2]), int(snr_roi[3]))

            noise_dict = dict(
                num_ims=1000,  # analyze this number of images at the end of the cleaned blinking images/video
                snr_roi=snr_roi,
                max_pv=max_pv,
            )

            mu, std, p, mpv = mu_std_p(param_dict, noise_dict)

            baseline = (np.round(1.0*mu), np.round(1.4*mu))
            read_std = (np.round(1.0*std), np.round(1.4*std))
            Nsig_range = (np.round(0.5*p/1e3)*1e3, np.round(1.1*p/1e3)*1e3)

            # update param_dict
            param_dict['baseline'] = baseline
            param_dict['read_std'] = read_std
            param_dict['Nsig_range'] = Nsig_range
            param_dict['bg'] = 0  # the param_dict in state will also be updated

            print(f'noise baseline: ({baseline[0]}, {baseline[1]})')
            print(f'noise std: ({read_std[0]}, {read_std[1]})')
            print(f'photon count: ({Nsig_range[0]}, {Nsig_range[1]})')
            return ('SNR is characterized.'
                    f'\nMPV: {mpv}.')


# training data generation
def func4(M, NA,  n_immersion, lamda, n_sample, f_4f, ps_camera, ps_BFP, external_mask,
          zstack_file, nfp_text, NFP, zrange, raw_image_folder, snr_roi, max_pv, projection_01,
          num_z_voxel, training_im_size, us_factor, max_num_particles, num_training_images, previous_param_dict, test_idx, threshold,
          state):  # training data

    if 'param_dict' not in state.keys():  # in the case of preprocessing images before characterizing PSF
        print('Cannot proceed.')
    else:
        # fetch param_dict
        param_dict = state['param_dict']
        if 'phase_mask' not in param_dict.keys():
            print('Cannot proceed. Please characterize PSF first.')
        else:
            # update param_dict
            param_dict['H'] = int(training_im_size)
            param_dict['W'] = int(training_im_size)
            param_dict['D'] = int(num_z_voxel)
            param_dict['us_factor'] = int(us_factor)
            param_dict['psf_half_size'] = 20  # pixels
            param_dict['num_particles_range'] = [1, int(max_num_particles)]
            param_dict['blob_r'] = 3 # changed on 17/03/2026 from 2 to prevent z coupeling
            param_dict['blob_sigma'] = 0.65  # changed on 03/04/2026 from 0.65 to prevent z coupeling
            param_dict['blob_maxv'] = 1000 * 5   # maximum value of network output

            param_dict['non_uniform_noise_flag'] = True

            param_dict['bitdepth'] = 16

            param_dict['HH'] = int(param_dict['H'] * us_factor)  # in case upsampling is needed
            param_dict['WW'] = int(param_dict['W'] * us_factor)
            param_dict['buffer_HH'] = int(param_dict['psf_half_size'] * us_factor)
            param_dict['buffer_WW'] = int(param_dict['psf_half_size'] * us_factor)

            param_dict['ps_xy'] = param_dict['ps_camera'] / param_dict['M']
            vs_xy = param_dict['ps_camera'] / param_dict['M'] / us_factor  # index of each voxel is at the center of the voxel
            vs_z = ((param_dict['zrange'][1] - param_dict['zrange'][0]) / param_dict['D'])   # no buffer zone in z axis
            param_dict['vs_xy'] = vs_xy
            param_dict['vs_z'] = vs_z

            param_dict['td_folder'] = os.path.join(os.getcwd(), 'training_data')  # where to save the training data
            if projection_01 == 0:
                param_dict['project_01'] = False  # seems better to not have 01 normalization
            else:
                param_dict['project_01'] = True

            param_dict['n_ims'] = int(num_training_images)  # the number of images for training

            '''
            # added on 21/07/2026 - free emitters in fov
            param_dict["free_emitters_bg_flag"] = False
            # Number of free emitters per full simulated frame, before tiling.
            param_dict["free_emitters_num_range"] = (1000, 3000)
            # Intensity range per free emitter.
            # Start relatively weak; tune visually using sim_exp.tif / training tiles.
            param_dict["free_emitters_photon_range"] = (2000.0, 6000.0)
            # Use same z range first.
            param_dict["free_emitters_z_range_um"] = param_dict["zrange"]
            # Isotropic diffusion blur in pixels.
            param_dict["free_emitters_gaussian_sigma_range_px"] = (5.0, 10.0)
            # Allow emitters just outside the FOV, so partial clouds appear at borders.
            param_dict["free_emitters_margin_px"] = 80.0
            # end 21/07/2026 - free emitters in fov
            '''
            training_data_func(param_dict)

            # show exp and sim together
            x_folder = os.path.join(param_dict['td_folder'], 'x')
            im_sim = io.imread(os.path.join(x_folder, os.listdir(x_folder)[0]))

            im_exp = io.imread(os.path.join(param_dict['im_br_folder'], os.listdir(param_dict['im_br_folder'])[0]))
            rr = max([im_sim.shape[0], im_exp.shape[0]])
            cc = max([im_sim.shape[1], im_exp.shape[1]])
            im_sim_exp = np.zeros((rr, 2*cc))
            im_sim_exp[:im_sim.shape[0], :im_sim.shape[1]] = im_sim
            im_sim_exp[:im_exp.shape[0], cc:cc+im_exp.shape[1]] = im_exp
            im_sim_exp = im_sim_exp.astype(np.uint16)
            io.imsave('sim_exp.tif', im_sim_exp, check_contrast=False)
            print(f'visual comparison between sim and exp: sim_exp.tif')

            param_dict['im_sim_exp'] = im_sim_exp

            return (f'Training data generation is done. '
                    f'\ndata folder: {param_dict["td_folder"]}.'
                    f'\ncheck sim_exp.tif to tune MPV if necessary.')
            '''# ori's edit
            #param_dict['mask_offset_in_um'] = 40000.0 * 1# or whatever value you want (um); 0.0 means legacy/no offset

            self.mask_offset_in_um = float(param_dict.get("mask_offset_in_um", 0.0))
            self.centralBeadCoordinates_pixel = list(param_dict.get("centralBeadCoordinates_pixel", [600, 600]))

            #param_dict['mask_offset_in_um'] = param_dict.get('mask_offset_in_um', 0.0)
            param_dict['num_tiles'] = 8//2 # 8//2
            param_dict['center_fraction'] = 0.75 # 0.85 #0.6 works 14/04/2026  # overlapping area parameter - how much of the field of view will be considered
            #end ori's edit'''

# training
def func5(M, NA,  n_immersion, lamda, n_sample, f_4f, ps_camera, ps_BFP, external_mask,
          zstack_file, nfp_text, NFP, zrange, raw_image_folder, snr_roi, max_pv, projection_01,
          num_z_voxel, training_im_size, us_factor, max_num_particles, num_training_images, previous_param_dict, test_idx, threshold,
          state):
    if 'param_dict' not in state.keys():  # in the case of preprocessing images before characterizing PSF
        print('Cannot proceed.')
    else:
        # fetch param_dict
        param_dict = state['param_dict']
        if 'phase_mask' not in param_dict.keys():
            print('Cannot proceed. Please characterize PSF first.')
        else:
            # update param_dict
            param_dict['path_save'] = os.path.join(os.getcwd(), 'training_results')

            training_dict = dict(
                batch_size=16*2,
                lr=0.005,
                num_epochs=30*1,

                # resume_net_file = 'last_net_08-03_12-35.pt' #add "last_net file to resume checkpoint. 'None' is default!
                resume_net_file = None  # added updated functionality. 'None' is default!
            )

            net_file, fit_file = training_func(param_dict, training_dict)
            param_dict['net_file'] = net_file
            param_dict['fit_file'] = fit_file

            # save param_dict
            param_file_name = 'param_dict_' + net_file[4:-3] + '.pickle'
            with open(param_file_name, 'wb') as handle:
                pickle.dump(param_dict, handle)
            print(f'A training file is saved as {param_file_name}')

            return (f'Training is done. '
                    f'\nresult folder: {param_dict["path_save"]}. '
                    f'\nA training file [{param_file_name}] is saved for future use.')


# inference
def func6_1(M, NA,  n_immersion, lamda, n_sample, f_4f, ps_camera, ps_BFP, external_mask,
          zstack_file, nfp_text, NFP, zrange, raw_image_folder, snr_roi, max_pv, projection_01,
          num_z_voxel, training_im_size, us_factor, max_num_particles, num_training_images, previous_param_dict, test_idx, threshold,
          state):

    if previous_param_dict != 'None':  # overwrite the param_dict
        with open(previous_param_dict, 'rb') as handle:
            state['param_dict'] = pickle.load(handle)
            print(f'The training file is loaded.')

    if 'param_dict' not in state.keys():  # ensure param_dict exists
        print('Cannot proceed. No training file.')
    else:
        # fetch param_dict
        param_dict = state['param_dict']
        if 'phase_mask' not in param_dict.keys():
            print('Cannot proceed. Please characterize PSF first.')
        else:
            image_br_folder = raw_image_folder + '_br'
            if not os.path.isdir(image_br_folder):
                print(f'Cannot proceed. Please preprocess images (background removal).')
            else:
                param_dict['im_br_folder'] = image_br_folder
                param_dict_copy = param_dict.copy()  # don't change the state['param_dict']
                param_dict_copy['threshold'] = threshold
                inference_func1(param_dict_copy, test_idx)
                return ('Test is done. check'
                        '\nloss_curves.jpg'
                        '\nsim_loc_gt_rec.jpg'
                        '\nsim_im_gt_rec.jpg'
                        '\nexp_im_gt_rec.jpg')


def func6_2(M, NA,  n_immersion, lamda, n_sample, f_4f, ps_camera, ps_BFP, external_mask,
          zstack_file, nfp_text, NFP, zrange, raw_image_folder, snr_roi, max_pv, projection_01,
          num_z_voxel, training_im_size, us_factor, max_num_particles, num_training_images, previous_param_dict, test_idx, threshold,
          state):

    if previous_param_dict != 'None':  # overwrite the param_dict
        with open(previous_param_dict, 'rb') as handle:
            state['param_dict'] = pickle.load(handle)
            print(f'The training file is loaded.')

    if 'param_dict' not in state.keys():  # in the case of preprocessing images before characterizing PSF
        print('Cannot proceed. No training file.')
    else:
        # fetch param_dict
        param_dict = state['param_dict']
        if 'phase_mask' not in param_dict.keys():
            print('Cannot proceed. Please characterize PSF first.')
        else:
            image_br_folder = raw_image_folder + '_br'
            if not os.path.isdir(image_br_folder):
                print(f'Cannot proceed. Please preprocess images (background removal).')
            else:
                param_dict['im_br_folder'] = image_br_folder
                param_dict_copy = param_dict.copy()  # don't change the state['param_dict']
                param_dict_copy['threshold'] = threshold
                file_name = inference_func2(param_dict_copy)
                return f'Obtained a localization list file: {file_name}.'


# one click
def func7(M, NA,  n_immersion, lamda, n_sample, f_4f, ps_camera, ps_BFP, external_mask,
          zstack_file, nfp_text, NFP, zrange, raw_image_folder, snr_roi, max_pv, projection_01,
          num_z_voxel, training_im_size, us_factor, max_num_particles, num_training_images, previous_param_dict, test_idx, threshold,
          state):

    func1(M, NA,  n_immersion, lamda, n_sample, f_4f, ps_camera, ps_BFP, external_mask,
          zstack_file, nfp_text, NFP, zrange, raw_image_folder, snr_roi, max_pv, projection_01,
          num_z_voxel, training_im_size, us_factor, max_num_particles, num_training_images, previous_param_dict, test_idx, threshold,
          state)
    func2(M, NA,  n_immersion, lamda, n_sample, f_4f, ps_camera, ps_BFP, external_mask,
          zstack_file, nfp_text, NFP, zrange, raw_image_folder, snr_roi, max_pv, projection_01,
          num_z_voxel, training_im_size, us_factor, max_num_particles, num_training_images, previous_param_dict, test_idx, threshold,
          state)
    func3(M, NA,  n_immersion, lamda, n_sample, f_4f, ps_camera, ps_BFP, external_mask,
          zstack_file, nfp_text, NFP, zrange, raw_image_folder, snr_roi, max_pv, projection_01,
          num_z_voxel, training_im_size, us_factor, max_num_particles, num_training_images, previous_param_dict, test_idx, threshold,
          state)
    func4(M, NA,  n_immersion, lamda, n_sample, f_4f, ps_camera, ps_BFP, external_mask,
          zstack_file, nfp_text, NFP, zrange, raw_image_folder, snr_roi, max_pv, projection_01,
          num_z_voxel, training_im_size, us_factor, max_num_particles, num_training_images, previous_param_dict, test_idx, threshold,
          state)
    func5(M, NA,  n_immersion, lamda, n_sample, f_4f, ps_camera, ps_BFP, external_mask,
          zstack_file, nfp_text, NFP, zrange, raw_image_folder, snr_roi, max_pv, projection_01,
          num_z_voxel, training_im_size, us_factor, max_num_particles, num_training_images, previous_param_dict, test_idx, threshold,
          state)
    func6_1(M, NA,  n_immersion, lamda, n_sample, f_4f, ps_camera, ps_BFP, external_mask,
          zstack_file, nfp_text, NFP, zrange, raw_image_folder, snr_roi, max_pv, projection_01,
          num_z_voxel, training_im_size, us_factor, max_num_particles, num_training_images, previous_param_dict, test_idx, threshold,
          state)
    func6_2(M, NA,  n_immersion, lamda, n_sample, f_4f, ps_camera, ps_BFP, external_mask,
          zstack_file, nfp_text, NFP, zrange, raw_image_folder, snr_roi, max_pv, projection_01,
          num_z_voxel, training_im_size, us_factor, max_num_particles, num_training_images, previous_param_dict, test_idx, threshold,
          state)

    return 'One click is done.'

