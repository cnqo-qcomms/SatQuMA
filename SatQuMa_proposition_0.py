# -*- coding: utf-8 -*-
"""
Created on Sun May 28 10:55:30 2023

@author: Sacha
"""

import numpy as np

def get_channel_efficiency(length):
    "η_ch = 10**(-0.2*L/10)"
    channel_efficiency = 10**(-0.2*length/10)
    return channel_efficiency

def get_binary_entropy(x : float ) -> float:
    binary_entropy = - x*np.log2(x) - (1 - x)*np.log2(1 - x)
    return binary_entropy

def gamma(a, b, c, d):
    left = (c + d)*(1 - b)*b/(c*d*np.log(2))
    right = np.log2((c + d)*(21**2)/(c*d*(1 - b)*b*a**2))
    gamma = np.sqrt(left*right)
    return gamma


class Key_Rate_Calculator():
    
    def __init__(self,
                 channel_efficiency : float,
                 number_of_pulses : float,
                 secrecy_error : float,
                 correctness_error : float,
                 average_observed_error_rate_in_x : float,
                 error_correction_efficiency : float,
                 intensity_1 : float,
                 intensity_2 : float,
                 intensity_3 : float,
                 intensisty_1_selection_probability : float,
                 intensisty_2_selection_probability : float,
                 x_basis_selection_probability : float,
                 receiver_efficiency, 
                 dark_count_probability, 
                 after_pulse_probability,
                 error_rate_due_to_optical_errors):
        
        if not (intensity_1 > intensity_2 + intensity_3):
            pass
        elif not (intensity_2 > intensity_3):
            pass
        elif not (intensity_3 >= 0):
            pass
        
        if not (0 <= intensisty_1_selection_probability <= 1):
            pass
        if not (0 <= intensisty_2_selection_probability <= 1):
            pass
        if not (0 <= x_basis_selection_probability <= 1):
            pass
        
        self.channel_efficiency = channel_efficiency
        self.number_of_pulses = number_of_pulses
        self.secrecy_error = secrecy_error
        self.correctness_error = correctness_error
        self.average_observed_error_rate_in_x = average_observed_error_rate_in_x
        self.error_correction_efficiency = error_correction_efficiency
        
        self.intensity_1 = intensity_1
        self.intensity_2 = intensity_2
        self.intensity_3 = intensity_3
        
        self.intensisty_1_selection_probability = intensisty_1_selection_probability
        self.intensisty_2_selection_probability = intensisty_2_selection_probability
        self.intensisty_3_selection_probability = self.get_intensity_selection_probability(intensity_3)
        
        self.x_basis_selection_probability = x_basis_selection_probability
        self.z_basis_selection_probability = self.get_basis_selection_probability("z")
        
        self.receiver_efficiency = receiver_efficiency
        self.dark_count_probability = dark_count_probability
        self.after_pulse_probability = after_pulse_probability
        self.error_rate_due_to_optical_errors = error_rate_due_to_optical_errors
        
    def get_intensity_selection_probability(self, intensity):
        if intensity == self.intensity_1:
            intensity_selection_probability =  self.intensisty_1_selection_probability
        elif intensity == self.intensity_2:
            intensity_selection_probability =  self.intensisty_2_selection_probability
        elif intensity == self.intensity_3:
            "p_μ3 = 1 - p_μ1  - p_μ2"
            intensity_selection_probability = (1 - self.intensisty_1_selection_probability
                                                 - self.intensisty_2_selection_probability)
        return intensity_selection_probability

    def get_basis_selection_probability(self, basis):
        "p_z = 1 - p_x"
        if basis == "x":
            return self.x_basis_selection_probability
        elif basis == "z":
            return 1 - self.x_basis_selection_probability
    
    def get_system_efficiency(self):
        "η_sys = η_ch*η_bob"
        system_efficiency = self.channel_efficiency*self.receiver_efficiency
        return system_efficiency
    
    def get_expecting_detection_rate_excluding_after_pulse(self, intensity):
        "D_k = 1 - (1 - 2*p_dc)*np.exp(-η_sys*k)"
        expecting_detection_rate_excluding_after_pulse = (1 - 
                                                         (1 - 2*self.dark_count_probability)*np.exp(-self.get_system_efficiency()*intensity))

        return expecting_detection_rate_excluding_after_pulse

    def get_expecting_detection_rate_including_after_pulse(self, intensity):
        "R_k = D_k*(1 + p_ap)"
        expecting_detection_rate_including_after_pulse = self.get_expecting_detection_rate_excluding_after_pulse(intensity)*(1 + self.after_pulse_probability)
        return expecting_detection_rate_including_after_pulse
        
    def get_bit_error_probability(self, intensity):
        "e_k = p_dc + e_mis*(1 - np.exp(-η_ch*k)) + 0.5*p_ap*D_k"
        bit_error_probability = (self.dark_count_probability
                                 + self.error_rate_due_to_optical_errors*(1 - np.exp(-self.channel_efficiency*intensity))
                                 + 0.5*self.after_pulse_probability*self.get_expecting_detection_rate_excluding_after_pulse(intensity))
        return bit_error_probability
    
    def get_number_of_events(self, intensity, basis):
        "n_j_k = ..."
        number_of_events = self.get_basis_selection_probability(basis)*self.get_basis_selection_probability(basis)*self.number_of_pulses*self.get_intensity_selection_probability(intensity)*self.get_expecting_detection_rate_excluding_after_pulse(intensity)
        return number_of_events
    
    def get_post_processing_block_size(self, basis):
        "n_j = ..."
        post_processing_block_size = (self.get_number_of_events(self.intensity_1, basis)
                                      + self.get_number_of_events(self.intensity_2, basis)
                                      + self.get_number_of_events(self.intensity_3, basis))
        return post_processing_block_size
    
    def get_number_of_events_upperbound(self, intensity, basis):
        "n_j_k+ = ..."
        upperbound = np.exp(intensity)*(self.get_number_of_events(intensity, basis) 
                                        + np.sqrt(0.5*self.get_post_processing_block_size(basis)*np.log(21/self.secrecy_error)))/self.get_intensity_selection_probability(intensity)
        return upperbound
    
    def get_number_of_events_lowerbound(self, intensity, basis):
        "n_j_k- = ..."
        lowerbound = np.exp(intensity)*(self.get_number_of_events(intensity, basis) 
                                        - np.sqrt(0.5*self.get_post_processing_block_size(basis)*np.log(21/self.secrecy_error)))/self.get_intensity_selection_probability(intensity)
        return lowerbound
    
    def get_number_of_errors(self, intensity, basis):
        "m_j_k = ..."
        number_of_errors = self.get_bit_error_probability(intensity)*self.get_number_of_events(intensity, basis)/self.get_expecting_detection_rate_excluding_after_pulse(intensity)
        return number_of_errors
    
    def get_post_processing_block_size_2(self, basis): # A renommer
        "m_j = ..."
        post_processing_block_size = (self.get_number_of_errors(self.intensity_1, basis)
                                      + self.get_number_of_errors(self.intensity_2, basis)
                                      + self.get_number_of_errors(self.intensity_3, basis))
        return post_processing_block_size
    
    def get_number_of_errors_upperbound(self, intensity, basis):
        "m_j_k+ = ..."
        upperbound = np.exp(intensity)*(self.get_number_of_errors(intensity, basis) 
                                        + np.sqrt(0.5*self.get_post_processing_block_size_2(basis)*np.log(21/self.secrecy_error)))/self.get_intensity_selection_probability(intensity)
        return upperbound
    
    def get_number_of_errors_lowerbound(self, intensity, basis):
        "m_j_k- = ..."
        lowerbound = np.exp(intensity)*(self.get_number_of_errors(intensity, basis) 
                                        - np.sqrt(0.5*self.get_post_processing_block_size_2(basis)*np.log(21/self.secrecy_error)))/self.get_intensity_selection_probability(intensity)
        return lowerbound
    
    def get_quantum_bit_error_rate(self, basis):
        "QBER = m_j/n_j"
        quantum_bit_error_rate = self.get_post_processing_block_size_2(basis)/self.get_post_processing_block_size(basis)
        return quantum_bit_error_rate

    def get_n_photon_state_transmission_probability(self, n):
        """
        τ_n =  ∑ (np.exp(-k)*p_k*k**n)/np.math.factorial(n), k ∈ K = {μ1, μ2, μ3}
        """
        n_photon_state_transmission_probability = ((np.exp(-self.intensity_1)*self.get_intensity_selection_probability(self.intensity_1)*self.intensity_1**n)
                                                    + (np.exp(-self.intensity_2)*self.get_intensity_selection_probability(self.intensity_2)*self.intensity_2**n)
                                                    + (np.exp(-self.intensity_3)*self.get_intensity_selection_probability(self.intensity_3)*self.intensity_3**n))/np.math.factorial(n)
        return n_photon_state_transmission_probability
    
    def get_appoximate_number_of_vacuum_events(self, basis):
        "s_j_0 = ..."
        numerator = (self.intensity_2*self.get_number_of_events_lowerbound(self.intensity_3, basis)
                      - self.intensity_3*self.get_number_of_events_upperbound(self.intensity_2, basis))*self.get_n_photon_state_transmission_probability(0)
        denominator = self.intensity_2 - self.intensity_3
        appoximate_number_of_vacuum_events = numerator/denominator
        return appoximate_number_of_vacuum_events
    
    def get_appoximate_number_of_single_photon_events(self, basis):
        "s_j_1 = ..."
        numerator = self.get_n_photon_state_transmission_probability(1)*self.intensity_1*(self.get_number_of_events_lowerbound(self.intensity_2, basis)
                                                                                          - self.get_number_of_events_upperbound(self.intensity_3, basis)
                                                                                          - (self.intensity_2**2 - self.intensity_3**2)*(self.get_number_of_events_upperbound(self.intensity_1, basis) 
                                                                                                                                          - self.get_appoximate_number_of_vacuum_events(basis)/self.get_n_photon_state_transmission_probability(0))/self.intensity_1**2)
        denominator = self.intensity_1*(self.intensity_2 - self.intensity_3) - self.intensity_2**2 + self.intensity_3**2
        appoximate_number_of_single_photon_events = numerator/denominator
        return appoximate_number_of_single_photon_events
    
    def get_approximate_number_of_bit_error(self, basis):
        "v_j_1 = ..."
        numerator = self.get_n_photon_state_transmission_probability(1)*(self.get_number_of_errors_upperbound(self.intensity_2, basis) - self.get_number_of_errors_lowerbound(self.intensity_3, basis))
        denominator = self.intensity_2 - self.intensity_3
        approximate_number_of_bit_error = numerator/denominator
        return approximate_number_of_bit_error
    
    def get_phase_error_rate(self):
        "φ_x = ..."
        phase_error_rate = (self.get_approximate_number_of_bit_error("z")/self.get_appoximate_number_of_single_photon_events("z")
                            + gamma(self.secrecy_error,
                                    self.get_approximate_number_of_bit_error("z")/self.get_appoximate_number_of_single_photon_events("z"),
                                    self.get_appoximate_number_of_single_photon_events("z"),
                                    self.get_appoximate_number_of_single_photon_events("x")))
        return phase_error_rate
    
    def get_estimated_number_of_bits_for_error_correction(self, basis):
        "leak_EC = ..."
        number_of_bits_for_error_correction = self.error_correction_efficiency*self.get_post_processing_block_size(basis)*get_binary_entropy(self.get_quantum_bit_error_rate(basis))
        return number_of_bits_for_error_correction
    
    def get_secret_key_length(self):
        "l = ..."
        secret_key_length = (self.get_appoximate_number_of_vacuum_events("x")
                              + self.get_appoximate_number_of_single_photon_events("x")
                              - self.get_appoximate_number_of_single_photon_events("x")*get_binary_entropy(self.get_phase_error_rate())
                              - self.get_estimated_number_of_bits_for_error_correction("x")
                              - 6*np.log2(21/self.secrecy_error)
                              - np.log2(2/self.correctness_error))
        return secret_key_length
    

import matplotlib.pyplot as plt
from collections import defaultdict

krc = Key_Rate_Calculator(channel_efficiency = 0.000194,
                          number_of_pulses = 1e9,
                          secrecy_error = 1e-9,
                          correctness_error = 1e-15,
                          average_observed_error_rate_in_x = 10,
                          error_correction_efficiency = 1.16,
                          intensity_1 = 0.7921,
                          intensity_2 = 0.1707,
                          intensity_3 = 0,
                          intensisty_1_selection_probability = 0.7501,
                          intensisty_2_selection_probability = 0.1749,
                          x_basis_selection_probability = 0.7611,
                          receiver_efficiency = 1.0, 
                          dark_count_probability = 1.00e-08, 
                          after_pulse_probability = 0.001,
                          error_rate_due_to_optical_errors = 0.001)
krc.get_secret_key_length()
        
# x = defaultdict(list) 
# y = defaultdict(list) 
# for s in range(4, 10):     
#     for l in list(range(201)):
#         krc = Key_Rate_Calculator(channel_efficiency = get_channel_efficiency(l),
#                                   number_of_pulses = 10**s,
#                                   secrecy_error = 1e-9,
#                                   correctness_error = 1e-15,
#                                   average_observed_error_rate_in_x = 10,
#                                   error_correction_efficiency = 1.16,
#                                   intensity_1 = 0.7921,
#                                   intensity_2 = 0.1707,
#                                   intensity_3 = 0,
#                                   intensisty_1_selection_probability = 0.7501,
#                                   intensisty_2_selection_probability = 0.1749,
#                                   x_basis_selection_probability = 0.7611,
#                                   receiver_efficiency = 1.0, 
#                                   dark_count_probability = 1.00e-08, 
#                                   after_pulse_probability = 0.001,
#                                   error_rate_due_to_optical_errors = 0.001)
#         skr = krc.get_secret_key_length()
#         if (not np.isnan(skr) and skr > 0):
#             x[s].append(l)
#             y[s].append(krc.get_secret_key_length())
#     plt.semilogy(x[s],  y[s])

    
  
    
    