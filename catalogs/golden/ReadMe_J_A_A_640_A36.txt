J/A+A/640/A36 OB stars TESS phot. & high-resolution spectroscopy (Bowman+, 2020)
================================================================================
Photometric detection of internal gravity waves in upper main-sequence stars.
II. Combined TESS photometry and high-resolution spectroscopy.
    Bowman D.M., Burssens S., Simon-diaz S., Edelmann P.V.F., Rogers T.M.,
    Horst L., Ropke F.K., Aerts C.
   <Astron. Astrophys., 640, A36 (2020)>
   =2020A&A...640A..36B    (SIMBAD/NED BibCode)
================================================================================
ADC_Keywords: Stars, OB ; Asteroseismology ; Spectroscopy
Keywords: asteroseismology - stars: early-type - stars: oscillations -
          stars: evolution - stars: rotation - stars: fundamental parameters

Abstract:
    Massive stars are predicted to excite internal gravity waves (IGWs) by
    turbulent core convection and from turbulent pressure fluctuations in
    their near-surface layers. These IGWs are extremely efficient at
    transporting angular momentum and chemical species within stellar
    interiors, but they remain largely unconstrained observationally.

    We aim to characterise the photometric detection of IGWs across a
    large number of O and early-B stars in the Hertzsprung-Russell
    diagram, and explain the ubiquitous detection of stochastic
    variability in the photospheres of massive stars.

    We combined high-precision time-series photometry from the NASA
    Transiting Exoplanet Survey Satellite with high-resolution
    ground-based spectroscopy of 70 stars with spectral types O and B to
    probe the relationship between the photometric signatures of IGWs and
    parameters such as spectroscopic mass, luminosity, and
    macroturbulence.

    A relationship is found between the location of a star in the
    spectroscopic Hertzsprung-Russell diagram and the amplitudes and
    frequencies of stochastic photometric variability in the light curves
    of massive stars. Furthermore, the properties of the stochastic
    variability are statistically correlated with macroturbulent velocity
    broadening in the spectral lines of massive stars.

    The common ensemble morphology for the stochastic low-frequency
    variability detected in space photometry and its relationship to
    macroturbulence is strong evidence for IGWs in massive stars, since
    these types of waves are unique in providing the dominant tangential
    velocity field required to explain the observed spectroscopy.

Description:
    The spectroscopic parameters of our sample are provided by Burssens et
    al. (2020A&A...639A..81B , Cat. J/A+A/639/A81) and include the
    effective temperature, Teff, spectroscopic luminosity,
    log10(Ls/Ls_{sun}_), projected surface rotational velocity,
    vsini, and macroturbulent broadening, vmacro. These were derived
    from high-resolution spectra assembled by the IACOB (Simon-Diaz et
    al., 2011, Bull. Soc. R. Sci. Liege, 80, 514; 2015, in Highlights of
    Spanish Astrophysics VIII, eds. A. J. Cenarro, F. Figueras, C.
    Hernandez-Monteagudo, J. Trujillo Bueno, & L. Valdivielso, 576) and
    OWN (Barba et al. 2010, Rev. Mex. Astron. Astrofis. Conf. Ser., 38,
    30; 2014, Rev. Mex. Astron. Astrofis. Conf. Ser., 44, 148; 2017, in
    The Lives and Death-Throes of Massive Stars, eds. J. J. Eldridge, J.
    C. Bray, L. A. S. McClelland, & L. Xiao, IAU Symp., 329, 89) surveys.

File Summary:
--------------------------------------------------------------------------------
 FileName      Lrecl  Records   Explanations
--------------------------------------------------------------------------------
ReadMe            80        .   This file
tablea1.dat       57       70   Parameters of OB stars studied in this work
tablea2.dat       83       70   Optimised parameters for the morphology of
                                 low-frequency variability (cf. Eq. (2)) using
                                 a Bayesian MCMC fitting method
--------------------------------------------------------------------------------

See also:
           IV/38 : TESS Input Catalog - v8.0 (TIC-8) (Stassun+, 2019)
   J/A+A/639/A81 : Variability of OB stars (Burssens+, 2020)

Byte-by-byte Description of file: tablea1.dat
--------------------------------------------------------------------------------
   Bytes Format Units  Label      Explanations
--------------------------------------------------------------------------------
       1  I1    ---    Cat        [1/9] Category (G1)
   3- 13  A11   ---    Name       Common name
  15- 23  I9    ---    TIC        TIC number
  25- 39  A15   ---    SpType     Spectral type
  41- 44  F4.2  [K]    logTeff    Effective temperature
                                   (typical uncertainty 0.03dex)
  46- 49  F4.2  [Sun]  logLs      Spectroscopic luminosity (where Ls=Teff^4^/g)
                                   (typical uncertainty 0.15dex)
  51- 53  I3    km/s   vsini      Projected surface rotational velocity (1)
  55- 57  I3    km/s   vmacro     ? Macroturbulent broadening (1)
--------------------------------------------------------------------------------
Note (1): taken from Burssens et al. (2020A&A...639A..81B, Cat. J/A+A/639/A81).
--------------------------------------------------------------------------------

Byte-by-byte Description of file: tablea2.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label     Explanations
--------------------------------------------------------------------------------
       1  I1    ---     Cat       [1/9] Category (G1)
   3- 13  A11   ---     Name      Common name
  15- 23  I9    --      TIC       TIC number
  25- 32  F8.3  umag    alpha0    Amplitude at a frequency of zero
  34- 38  F5.3  umag  e_alpha0    rms uncertainty on alpha0
  40- 46  F7.5  d-1     nuchar    Characteristic frequency
  48- 54  F7.5  d-1   e_nuchar    rms uncertainty on nuchar
  56- 62  F7.5  ---     gamma     logarithmic amplitude gradient
  64- 70  F7.5  ---   e_gamma     rms uncertainty on gamma
  72- 77  F6.3  umag    Cw        Frequency-independent noise term
  79- 83  F5.3  umag  e_Cw        rms uncertainty on Cw
--------------------------------------------------------------------------------

Global notes:
Note (G1): Category as follows:
            1 = O dwarf stars
            2 = O subgiant stars
            3 = O giant stars
            4 = O bright giant and supergiant stars
            5 = B dwarf stars
            6 = B subgiant stars
            7 = B giant stars
            8 = B bright giant and supergiant stars
            9 = Peculiar stars
--------------------------------------------------------------------------------

History:
    From electronic version of the journal

References:
    Bowman et al., Paper I   2019A&A...621A.135B

================================================================================
(End)                                      Patricia Vannier [CDS]    09-Oct-2020
