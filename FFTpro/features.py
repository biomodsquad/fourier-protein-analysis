from math import log2, ceil
import numpy as np
import pandas as pd
from scipy.fft import fft, ifft

from FFTpro.utility import frequency_select_physical
from FFTpro.embeddings import b100

class SequenceFFT():
    """Fourier transform encoding of a protein/nucleotide sequence using amino 
    acids/nucleotide embeddings.

    Parameters
    ----------
    sequence : str
        Protein/nucleotide sequence in one-letter code.
    
    attributes : dict
        Additional sequence attributes.
    """
       
    def __init__(self, sequence, attributes = None):
        self.sequence = sequence

        if attributes is not None:
            for name, value in attributes.items():
                setattr(self, name, value)

    def embed_sequence(self, embeddings):
        """Embed protein/nucleotide sequence. 

        Updates SequenceFFT and adds embeddings (dict) attribute.

        Parameters
        ----------
        embeddings: dict of dict 
            Each item of the dictionary should be a dictionary containing items for each 
            amino acid/nucleotide named in single letter code with numberical embedding.
            See b100 (BLOSSUM100) for an example.

        """
        
        self.embeddings = {}
        for embedding in embeddings.keys():
            self.embeddings[embedding] = [embeddings[embedding][amino_acid] for amino_acid in self.sequence]
            
    def compute_spectra(self,length):
        """Compute Fast Fourier Transforms for each sequence embedding. 

        Updates SequenceFFT and adds spectra (dict) and spectrum_length (int) attributes.
        The items of the spectra dictionary are named a according to the names of the 
        embeddings.

        Parameters
        ----------
        length: int 
            Length of spectra. Should be at least as long as sequence but longer lengths
            can be achieved with zero padding. 

        """
        
        self.frequency = np.linspace(-0.5,0.5, num = length+1, endpoint = True)
        self.spectrum_length = length
        N = len(self.sequence)
        
        self.spectra = {}
        for embedding in self.embeddings.keys():
            padded_vector = np.zeros(self.spectrum_length)
            padded_vector[range(0,N)] = self.embeddings[embedding]
        
            fft_spectrum = fft(padded_vector)  
            fft_spectrum = np.append(fft_spectrum,fft_spectrum[0])
            self.spectra[embedding] = fft_spectrum

    def compute_cross_spectra(self):
        """Compute cross-spectra between all FFT spectra. 

        Updates SequenceFFT and adds cross_spectra (dict) with names Embedding-1_Embedding-2. 

        """
        
        spectra_names = list(self.spectra.keys())
        
        self.cross_spectra = {}
        for i in range(len(spectra_names)):
            for j in range(i,len(spectra_names)):
                cross_name = spectra_names[i]+'_'+spectra_names[j]
                self.cross_spectra[cross_name] = self.spectra[spectra_names[i]]*np.conjugate(self.spectra[spectra_names[j]])

    def compute_spectral_properties(self, 
                                    spectra_types = ['spectra'], 
                                    select_method = frequency_select_physical, 
                                    compute_amplitude = True, 
                                    compute_phase = False):
        
        """Compute spectral properties of spectra and/or cross-spectra. 

        Parameters
        ----------
        spectra_type: list 
            List of spectra types (i.e. spectra or cross-spectra) for which to compute 
            properties. 
        
        select_method: callable 
            Function that takes frequency as input and returns bool identifying the desired
            frequency components for which to generate properties. 
            See utility.frequency_select_physical. 

        compute_amplitude: bool 
            Boolean indicating whether to compute amplitudes of frequency components. 

        compute_phase: bool 
            Boolean indicating whether to compute phase (angles) of frequency components. 

        Returns
        -------
        dict
            Dictionary of spectral properties named as 'Property_Spectrum_Frequency'.

        """
        
        N = len(self.sequence)
        sel_components = select_method(self.frequency)
        
        spectra_properties = {}
        for type in spectra_types:
 
            spectra = getattr(self, type)
            for spectrum in spectra.keys():
 
                if compute_amplitude:
                    fft_amplitudes = abs(spectra[spectrum])#/N
                    selected_amplitudes = fft_amplitudes[sel_components]
                    amplitude_names = ["Amp_" + spectrum + "_" + str(f) for f in self.frequency[sel_components]]
                    for name,amplitude in zip(amplitude_names,selected_amplitudes):
                        spectra_properties[name] = amplitude
        
                if compute_phase:
                    fft_phases = np.angle(spectra[spectrum])
                    selected_phases = fft_phases[sel_components]
                    phase_names = ["Phase_" + spectrum + "_" + str(f) for f in self.frequency[sel_components]]
                    for name,phase in zip(phase_names,selected_phases):
                        spectra_properties[name] = phase
                        
        return spectra_properties 
            

class SequenceSet():
    def __init__(self, embedding = b100, compute_cross_spectra = True):
        self.embedding = embedding
        self.compute_cross_spectra = compute_cross_spectra

    def from_sequence_list(self, sequences, spectrum_length = None, attributes = None):
        self.sequences = sequences
        self.monomers = []
        
        if spectrum_length is None:
            sequence_lengths = [len(seq) for seq in self.sequences]
            max_length = max(sequence_lengths)
            self.spectrum_length = 2**ceil(log2(max_length))
        else:
            self.spectrum_length = spectrum_length
        
        if attributes is not None:
            attributes_list = [dict(zip(attributes, i)) for i in zip(*attributes.values())]
        else:
            attributes_list = [None]*len(self.sequences)
            
        for sequence,sequence_attributes in zip(sequences,attributes_list):
            sequence_fft = SequenceFFT(sequence, sequence_attributes)
            sequence_fft.embed_sequence(self.embedding)
            sequence_fft.compute_spectra(self.spectrum_length)
            
            if self.compute_cross_spectra:
                sequence_fft.compute_cross_spectra()
            
            self.monomers.append(sequence_fft)

    def export_datatable(self, 
                         spectra = ['spectra'], 
                         attributes = None, 
                         select_method = frequency_select_physical, 
                         compute_amplitude = True, 
                         compute_phase = False):
        set_metrics = []
        for monomer in self.monomers:
            monomer_metrics = {}
            if attributes is not None:
                for attribute in attributes:
                    monomer_metrics[attribute] = getattr(monomer,attribute)
                    
            monomer_metrics.update(monomer.compute_spectral_properties(spectra, select_method, 
                                                                       compute_amplitude, compute_phase))
            
            set_metrics.append(monomer_metrics)

        return pd.DataFrame(set_metrics)


class ComplexSet():
    def __init__(self, embedding = b100, compute_cross_cross_spectra = False):
        self.embedding = embedding
        self.compute_cross_cross_spectra = compute_cross_cross_spectra

    def from_sequence_list(self, A_sequences, B_sequences, spectrum_length = None, attributes = None):
        self.A_sequences = A_sequences
        self.B_sequences = B_sequences
        
        self.A = []
        self.B = []
        self.AB = []
        
        if spectrum_length is None:
            A_sequence_lengths = [len(seq) for seq in self.A_sequences]
            B_sequence_lengths = [len(seq) for seq in self.B_sequences]
            max_length = max((max(A_sequence_lengths),max(B_sequence_lengths)))
            self.spectrum_length = 2**ceil(log2(max_length))
        else:
            self.spectrum_length = spectrum_length
        
        if attributes is not None:
            attributes_list = [dict(zip(attributes, i)) for i in zip(*attributes.values())]
        else:
            attributes_list = [None]*len(A_sequences)
             
        for A_sequence,B_sequence,complex_attributes in zip(A_sequences,B_sequences,attributes_list):
            A_fft = SequenceFFT(A_sequence, complex_attributes)
            A_fft.embed_sequence(self.embedding)
            A_fft.compute_spectra(self.spectrum_length)
            
            B_fft = SequenceFFT(B_sequence, complex_attributes)
            B_fft.embed_sequence(self.embedding)
            B_fft.compute_spectra(self.spectrum_length)

            AB_fft = SequenceFFT([A_sequence,B_sequence], complex_attributes)
            AB_fft.frequency = A_fft.frequency
            AB_fft.spectrum_length = A_fft.spectrum_length

            A_spectra_names = list(A_fft.spectra.keys())
            B_spectra_names = list(B_fft.spectra.keys())
            
            AB_fft.spectra = {}
            for i in range(len(A_spectra_names)):
                for j in range(len(B_spectra_names)):
                    AB_spectra_name = 'A-'+A_spectra_names[i]+'_'+'B-'+B_spectra_names[j]
                    AB_fft.spectra[AB_spectra_name] = A_fft.spectra[A_spectra_names[i]]*\
                                                        np.conjugate(B_fft.spectra[B_spectra_names[j]])
            
            self.A.append(A_fft)
            self.B.append(B_fft)
            self.AB.append(AB_fft)
       

    def export_datatable(self, 
                         chain_ID = 'AB', 
                         attributes = None, 
                         select_method = frequency_select_physical, 
                         compute_amplitude = True, 
                         compute_phase = False):
        
        chains = getattr(self,chain_ID)
        
        set_metrics = []
        for chain in chains:
            chain_metrics = {}
            if attributes is not None:
                for attribute in attributes:
                    chain_metrics[attribute] = getattr(chain,attribute)
                    
            chain_metrics.update(chain.compute_spectral_properties(['spectra'], select_method, 
                                                                compute_amplitude, compute_phase))
            
            set_metrics.append(chain_metrics)

        return pd.DataFrame(set_metrics)


