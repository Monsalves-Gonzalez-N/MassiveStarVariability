J/A+A/639/A81         Variability of OB stars                  (Burssens+, 2020)
================================================================================
Variability of OB stars from TESS southern Sectors 1-13 and high-resolution
IACOB and OWN spectroscopy.
    Burssens S., Simon-Diaz S., Bowman D.M., Holgado G., Michielsen M.,
    De Burgos A., Castro N., Barba R.H., Aerts C.
   <Astron. Astrophys., 639, A81 (2020)>
   =2020A&A...639A..81B    (SIMBAD/NED BibCode)
================================================================================
ADC_Keywords: Stars, OB ; Asteroseismology ; Spectroscopy
Keywords: techniques: photometric - techniques: spectroscopic - stars: massive -
          stars: oscillations

Abstract:
    The lack of high-precision long-term continuous photometric data for
    large samples of stars has impeded the large-scale exploration of
    pulsational variability in the OB star regime. As a result, the
    candidates for in-depth asteroseismic modelling have remained limited
    to a few dozen dwarfs. The TESS nominal space mission has surveyed the
    southern sky, including parts of the galactic plane, yielding
    continuous data across at least 27d for hundreds of OB stars.

    We aim to couple TESS data in the southern sky with ground-based
    spectroscopy to study the variability in two dimensions, mass and
    evolution. We focus mainly on the presence of coherent pulsation modes
    that may or may not be present in the predicted theoretical
    instability domains and unravel all frequency behaviour in the
    amplitude spectra of the TESS data.

    We compose a sample of 98 OB-type stars observed by TESS in Sectors
    1-13 and with available multi-epoch, high-resolution spectroscopy
    gathered by the IACOB and OWN surveys. We present the short-cadence 2
    min light curves of dozens of OB-type stars, which have one or more
    spectra in the IACOB or OWN database. Based on these light curves and
    their Lomb-Scargle periodograms, we performed variability
    classification and frequency analysis. We placed the stars in the
    spectroscopic Hertzsprung-Russell diagram to interpret the variability
    in an evolutionary context.

    We deduce the diverse origins of the mmag-level variability found in
    all of the 98 OB stars in the TESS data. We find among the sample
    several new variable stars, including three hybrid pulsators, three
    eclipsing binaries, high frequency modes in a Be star, and potential
    heat-driven pulsations in two Oe stars.

    We identify stars for which future asteroseismic modelling is
    possible, provided mode identification is achieved. By comparing the
    position of the variables to theoretical instability strips, we
    discuss the current shortcomings in non-adiabatic pulsation theory and
    the distribution of pulsators in the upper Hertzsprung-Russell
    diagram.

Description:
    By combining TESS data and multi-epoch high-resolution spectroscopy
    gathered by the IACOB and OWN surveys, we provided a variability study
    for a sample of OB-type stars in the southern hemisphere based on
    visual inspections followed by a frequency analysis.

File Summary:
--------------------------------------------------------------------------------
 FileName      Lrecl  Records   Explanations
--------------------------------------------------------------------------------
ReadMe            80        .   This file
table2.dat        95       98   Overview of OB-type stars with available TESS
                                 photometry and IACOB/OWN spectroscopy with
                                 spectral type O4 to B3 considered in this work
tablea1.dat       97       98   Summary of the frequency analysis of the stars
                                 considered in this work
tablea2.dat       79      314   Detailed frequency analysis of multiperiodic
                                 pulsators for 26 stars
--------------------------------------------------------------------------------

See also:
    IV/38 : TESS Input Catalog - v8.0 (TIC-8) (Stassun+, 2019)

Byte-by-byte Description of file: table2.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label      Explanations
--------------------------------------------------------------------------------
       1  I1    ---     Sample     Sample code (G1)
   3- 13  A11   ---     Name       Star name
  15- 35  A21   ---     SpType     Spectral type
  37- 51  A15   ---     SpClass    Spectral class
  53- 56  F4.2  [K]     logTeff    ? Effective temperature
  58- 61  F4.2  [Sun]   logLs      ? Spectroscopic luminosity
                                    (where Ls=Teff^4^/g) (2)
  63- 65  I3    km/s    vsini      ? Rotational velocity
  67- 69  I3    km/s    vmac       ? Macroturbulent velocity
  71- 73  I3    ---     Nsp        Number of available spectra
  75- 82  A8    ---     VarTypesp  Spectral variability type (G3)
  84- 95  A12   ---     VarTypeT   TESS variability type (4)
--------------------------------------------------------------------------------
Note (2): computed using gravities already corrected for centrifugal
  acceleration (following the recipe proposed in
  Herrero et al., 1992A&A...261..209H and Repolust et al., 2004A&A...415..349R).
Note (4): TESS Variability types as follows:
  EB       = eclipsing binary
  rot      = rotational modulation
  SPB      = low frequency pulsation modes
  beta Cep = high frequency pulsation modes
  SLF      = stochastic low frequency signal
  PQ       = poor quality data
  cont.    = contaminated

  A question mark indicates that the TESS light curve is insufficient to
    disentangle the contribution of g modes, rotation effects and SLF.
--------------------------------------------------------------------------------

Byte-by-byte Description of file: tablea1.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label     Explanations
--------------------------------------------------------------------------------
       1  I1    ---     Sample    Sample code (G1)
   3- 13  A11   ---     Name      Star Name
  17- 25  I9    ---     TIC       TIC identification number
      27  A1    ---   n_TIC       [+] Note on TIC (2)
  29- 33  F5.3  d-1     1/DT      Frequency resolution of the amplitude spectra,
                                   defined as the reciprocal of the time-span
  35- 37  F3.1  d-1     Window    Noise window
  39- 46  F8.5  d-1     nudom     Dominant frequency
  49- 55  F7.5  d-1   e_nudom     rms uncertainty on nudom
  57- 63  F7.3  mmag    Adom      Amplitude of dominant frequency
  65- 69  F5.3  mmag  e_Adom      rms uncertainty on Adom
  71- 75  F5.2  ---     S/N       Signal-to-noise ratio
  77- 89  A13   ---     VarTypesp Spectral variability type (G3)
  91- 97  A7    ---     Notes     Notes (4)
--------------------------------------------------------------------------------
Note (2): + indicates that the light curve was extracted using our own method.
Note (4): Notes as follows:
 IV   = if the light curve is of poor quality or contaminated (invalid)
 HARM = if periodogram is dominated by harmonics
 *    = if more than one frequency was measured
--------------------------------------------------------------------------------

Byte-by-byte Description of file: tablea2.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label     Explanations
--------------------------------------------------------------------------------
       1  A1    ---     Sample    [A-F] Sample code (1)
   3- 13  A11   ---     Name      Star name
  14- 17  A4    ---     FreqName  Frequency designation
  21- 27  F7.4  d-1     Freq      Frequency value
  29- 34  F6.4  d-1   e_Freq      rms uncertainty on Freq
  37- 43  F7.3  mmag    Amp       Amplitude of variation of Freq
  45- 49  F5.3  mmag  e_Amp       rms uncertainty on Amp
  51- 56  F6.3  ---     phi       [-0.5/0.5] Phase
  58- 62  F5.3  ---   e_phi       rms uncertainty on phi
  65- 68  F4.1  ---     S/N       Signal-to-noise ratio
  71- 79  A9    ---     Notes     Notes (2)
--------------------------------------------------------------------------------
Note (1): Sample as follows:
  A = beta Cep stars
  B = SPB stars
  C = Hybrid stars
  D = Be/Oe stars
  E = Magnetic stars
  F = Eclipsing binaries
Note (2): nuorb indicates the orbital frequency, which is not always directly
  measured but derived from harmonics. In the case of HD 47129, nuorb refers to
  the orbital frequency measured by Mahy et al. (2011A&A...525A.101M).
  Potential combinations, harmonics or multiplet memberships are indicates as
  tp/tp? (triplet) or mp/mp? (multiplet) followed by a number if more than one
  was detected. See Appendix B for more details.
--------------------------------------------------------------------------------

Global notes:
Note (G1): Sample code as follows
  1 = O-type dwarfs and subgiants (V and IV)
  2 = O-type giants, bright giants and supergiants (III, II and I)
  3 = Early B-type dwarfs, subgiants and giants (V, IV, and III)
  4 = B-type giants, bright giants and supergiants (III, II, and I)
  5 = Magnetic O- and B-type stars
  6 = Oe and Be stars
Note (G3): Spectroscopic Variability types as follows:
  SB1 = single lined spectroscopic binary
  SB2 = double lined spectroscopic binary
  LPV = line profile variability in photospheric lines
  WVa = Variability of the H{alpha} line in absorption
  WVe = Variability of the H{alpha} line in emission
--------------------------------------------------------------------------------

History:
    From electronic version of the journal

================================================================================
(End)                                      Patricia Vannier [CDS]    17-Sep-2020
