# intersubject.py: Module contains an Intersubject class that provides
# methods for computing one-sample and within-between 
# intersubject correlation and intersubject functional correlation

import numpy as np
# package modules relative import
from .tools import save_data # local custom module with useful functions; necessary for this code
from .basic_stats import pairwise_r, r_mean

# class for computing whole sample and witihin-between ISFC and ISC.
class Intersubject:
    def __init__(self, data_path, datasize):
        self.data_path = data_path
        self.datasize = datasize
        self.dims = (datasize[0] * datasize[1] * datasize[2], datasize[3]) # for quick matrix size
        self.subject_ids = {} # can accomodate either one or multiple ID groups
        self.group_mask = None
        self.isfc = {}
        self.isc = {}
        self._data_sum = {}
        self._voxel_sum = {}

        # self._subject_data = None
        # self._subject_mask = None


    def _get_sum(self, label):
        # sub_list, file_dir, datasize):

        # Function to get sum data in preparation for computing the average 
        # for one-to-average/leave-one-out ISC analysis
        # 
        # Make empty summed data output arrays
        sum_data = np.zeros(shape=(self.dims))
        sum_vox = np.zeros(shape=(self.dims[0]))
        _group_mask = np.zeros(shape=(self.dims[0]))

        # Save each subject's data and voxel mask
        sub_list = self.subject_ids[label]
        # print(sub_list)
        for sub in sub_list:
            # Load subject data from npz dict object
            subject_dict = np.load(self.data_path.format(sub))
            mask = subject_dict['mask']

            # Get subject's masked data
            data = np.zeros(shape=(len(mask), self.dims[1])) # matrix of zeros
            data[mask, :] = subject_dict['data'] 

            # Add subject data to the sample-wide sum array
            sum_data = sum_data + data
            sum_vox = sum_vox + mask
        
        assert sum_data.shape == self.dims, f"sum data shape is {sum_data.shape} but should be {self.dims}"
        assert sum_vox.shape == (self.dims[0], ), f"sum vox shape is {sum_vox.shape} but should be {self.dims[0]}"

        # Create boolean mask of sample-wide surviving voxels
        _group_mask = sum_vox.copy() > (0.7 * len(sub_list)) # True if 70% participants voxels survive; magic number from YC's!
        
        # Find union between group masks if already defined
        if self.group_mask is not None:
            self.group_mask = np.logical_or(self.group_mask, _group_mask)
        else:
            self.group_mask = _group_mask

        # Reduce summed data and voxels with mask
        self._data_sum[label] = sum_data[_group_mask, :]
        self._voxel_sum[label] = sum_vox[_group_mask]

        # return sum_data, sum_vox

        # Return summed data and summed voxel masks    
        # return (sum_data, sum_vox, all_kept_vox)

    # Calculates one subject's ISFC and ISC
    def _isfc_oneavg(self, sub_id, label, compare_method = 'entire', minus_one = True):

        # subject_data, subject_vox, summed_data, summed_vox, 
        # allkeptvox, sub_datasize, sum_datasize, 
        # compare_method='entire_sample', minus_one=True, isc_only=False
        # ):
        

        # Recursively calculate subject's isfc for within and between group method
        if compare_method == 'within_between':

            # Save dicts of within and between data separately from arguments
            label_list = list(self.subject_ids)
            
            # treat provided label as "within group" label
            within_isfc = self._isfc_oneavg(sub_id, label)

            # use remaining label as "between group"
            label_list.remove(label)
            label = label_list[0]

            between_isfc = self._isfc_oneavg(sub_id, label, minus_one=False)
            
            return within_isfc, between_isfc
        
        elif compare_method == 'entire':
            # assert summed_data.shape==(64, 10), summed_data.shape
            # assert summed_vox.shape==(64,), summed_vox.shape
            # Retrieve subject data with their mask
            subject_dict = np.load(self.data_path.format(sub_id))
            mask = subject_dict['mask']
            data = np.full(shape=(mask.shape[0], self.dims[1]), fill_value=np.nan)
            data[mask, :] = subject_dict['data']

            # Mask the subject's data and voxel mask with sample-wide voxel mask (potentially more restrictive)
            data = data[self.group_mask, :]
            mask = mask[self.group_mask]

            assert data.shape == self._data_sum[label].shape, f"subject data {data.shape} and data sum {self._data_sum[label].shape} have mismatched shapes"
            assert mask.shape == self._voxel_sum[label].shape, f"subject mask {mask.shape} and voxel sum {self._voxel_sum[label].shape} have mismatched shapes"
            
            # Check whether to subtract of subject data from summed data
            if minus_one==False:
                temp_avg = self._data_sum[label] / self._voxel_sum[label].reshape((self._voxel_sum[label].shape[0], 1))
            else:
            # Create sample-wide average timecourse minus subject's timecourse for all voxels
                # assert self._data_sum[label].shape == data.shape, f""
                # assert self._voxel_sum[label].shape == mask.shape, 

                numer = self._data_sum[label] - data # removes subject's data
                denom = self._voxel_sum[label] - mask # removes subject's voxels
                temp_avg = numer / denom.reshape((denom.shape[0], 1))

            # Compute ISFC correlation between subject and average-minus-one
            isfc_matrix = pairwise_r(data, temp_avg)

            # Check whether to return full ISFC matrix or ISC diagonal
            # if isc_only == True:
            #     isc = isfc_matrix.diagonal()
            #     return isc
            # Return subject x average ISFC matrix for all voxels
            # else:
            return isfc_matrix

    # group isfc
    def group_isfc(self, group_ids, save_subject=False, compare_method = "entire", keep_isfc=True, keep_isc=True):
        assert keep_isfc or keep_isc, "Either keep_isfc or keep_isc must be True"

        # Forces subject_id as dict if list is provided instead
        if type(group_ids) is list:
            # try:
            if any(isinstance(item, list) for item in group_ids):
                raise TypeError("'subject_id' cannot be nested lists; provide multiple lists as dict with simple keys (eg., 0, 1) instead")
            else:
                self.subject_ids['sample'] = group_ids
                assert type(self.subject_ids) is dict
            # except TypeError:
            #     print()
        else:
            self.subject_ids = group_ids

        label_list = list(self.subject_ids)
        # print('checking initial subject id inputs')
        # print('label list', label_list)
        # print('dict value type', type(self.subject_ids[label_list[0]]))
        # print('dict value', self.subject_ids[label_list[0]])

        # Empty array containers for final group statistics
        # >isfc shape is (voxel x voxel x subjects)
        # >isc shape is (voxel x subjects)
        def get_container(this_label=None, last_dim=None):
            # Function to generate an "empty" array to be filled afterward.
            # Note: currently only works for the Intersubject class. 

            # assert type(this_label) is str, "this_label must be str type"
            # assert type(last_dim) is int, "last_dim_sum must be int type"
            if this_label is not None:
                last_dim = len(self.subject_ids[this_label])

            isfc_container = np.full((self.dims[0], self.dims[0], last_dim), np.nan)
            isc_container = np.full((self.dims[0], last_dim), np.nan)
            return isfc_container, isc_container

        # Whole-sample ISFC/ISC
        if compare_method == "entire":
            
            # treat first group as the primary label
            label = label_list[0]

            # get containers
            self.isfc['entire'], self.isc['entire'] = get_container(label)
            
            # self.isfc['entire'] = np.full((self.dims[0], self.dims[0], len(self.subject_ids[label])), np.nan)
            # self.isc['entire'] = np.full((self.dims[0], len(self.subject_ids[label])), np.nan)

            # Get summed data for later averaging
            self._get_sum(label)

            # Get isfc/isc
            for i, sub in enumerate(self.subject_ids[label]):
                this_isfc = self._isfc_oneavg(sub, label, compare_method=compare_method)
                assert this_isfc.shape == self.isfc['entire'][:,:,0].shape, f"subject isfc {this_isfc.shape} and isfc container {self.isfc['entire'][:,:,0].shape} are mismatched"
                                    
                # if keep_isfc:
                self.isfc['entire'][:, :, i] = this_isfc
                # self._isfc = subject_isfc

                # if keep_isc
                self.isc['entire'][:, i] = this_isfc.diagonal() 
                # self._isc = subject_isfc.diagonal()

        # Within-between ISFC/ISC
        elif compare_method == "within_between":

            label_left, label_right = label_list[0], label_list[1]
            all_ids = self.subject_ids[label_left] + self.subject_ids[label_right]

            # Save array containers to attribute's dict
            self.isfc['within'], self.isc['within'] = get_container(last_dim = len(all_ids))
            self.isfc['between'], self.isc['between'] = get_container(last_dim = len(all_ids))

            # Get sums for both subject groups
            self._get_sum(label_left)
            self._get_sum(label_right)

            # Get within and betweeen isfc/isc
            for i, sub in enumerate(all_ids):
                # Treat label as 'within' group if true
                if sub in self.subject_ids[label_left]:
                    wb_isfc = self._isfc_oneavg(sub, label_left, compare_method=compare_method)

                elif sub in self.subject_ids[label_right]:
                    wb_isfc = self._isfc_oneavg(sub, label_right, compare_method=compare_method)
                
                self.isfc['within'][:,:,i] = wb_isfc[0]
                self.isfc['between'][:,:,i] = wb_isfc[1]
                self.isc['within'][:,i] = wb_isfc[0].diagonal()
                self.isc['between'][:,i] = wb_isfc[1].diagonal()
                
            # return 
        