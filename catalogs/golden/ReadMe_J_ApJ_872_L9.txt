J/ApJ/872/L9        TESS obs. of massive O and B stars        (Pedersen+, 2019)
================================================================================
Diverse variability of O and B stars revealed from 2-minute cadence light curves
in sectors 1 and 2 of the TESS mission: selection of an asteroseismic sample.
    Pedersen M.G., Chowdhury S., Johnston C., Bowman D.M., Aerts C., Handler G.,
    De Cat P., Neiner C., David-Uraz A., Buzasi D., Tkachenko A., Simon-Diaz S.,
    Moravveji E., Sikora J., Mirouh G.M., Lovekin C.C., Cantiello M.,
    Daszynska-Daszkiewicz J., Pigulski A., Vanderspek R.K., Ricker G.R.
   <Astrophys. J., 872, L9 (2019)>
   =2019ApJ...872L...9P
================================================================================
ADC_Keywords: Stars, OB; Magnitudes, absolute; Stars, variable; Photometry;
              Optical
Keywords: asteroseismology ; binaries: general ; stars: evolution ;
          stars: massive ; stars: oscillations (including pulsations) ;
          stars: rotation

Abstract:
    Uncertainties in stellar structure and evolution theory are largest
    for stars undergoing core convection on the main sequence. A powerful
    way to calibrate the free parameters used in the theory of stellar
    interiors is asteroseismology, which provides direct measurements of
    angular momentum and element transport. We report the detection and
    classification of new variable O and B stars using high-precision
    short-cadence (2 minutes) photometric observations assembled by the
    Transiting Exoplanet Survey Satellite (TESS). In our sample of 154 O
    and B stars, we detect a high percentage (90%) of variability. Among
    these we find 23 multiperiodic pulsators, 6 eclipsing binaries, 21
    rotational variables, and 25 stars with stochastic low-frequency
    variability. Several additional variables overlap between these
    categories. Our study of O and B stars not only demonstrates the high
    data quality achieved by TESS for optimal studies of the variability
    of the most massive stars in the universe, but also represents the
    first step toward the selection and composition of a large sample of O
    and B pulsators with high potential for joint asteroseismic and
    spectroscopic modeling of their interior structure with unprecedented
    precision.

Description:
    To study the variability of massive stars, we analyze a sample of 154
    O and B stars observed by the Transiting Exoplanet Survey Satellite
    (TESS) with 2 minutes cadence, of which 40 are located in the Large
    Magellanic Cloud (LMC).

    The data treated here were obtained by TESS in Sectors 1 (2018 July 25
    - August 22) and 2 (2018 August 23 - September 20) and are publicly
    available via the Mikulski Archive for Space Telescopes (MAST).

File Summary:
--------------------------------------------------------------------------------
 FileName    Lrecl  Records  Explanations
--------------------------------------------------------------------------------
ReadMe          80        .  This file
table1.dat     140      154  Identification numbers, parameters and variability
                              classification  of the 154 O and B stars
--------------------------------------------------------------------------------

See also:
 B/eso : ESO Science Archive Catalog (ESO, 1991-2020)
 I/345 : Gaia DR2 (Gaia Collaboration, 2018)
 I/347 : Distances to 1.33 billion stars in Gaia DR2 (Bailer-Jones+, 2018)
 IV/38 : TESS Input Catalog - v8.0 (TIC-8) (Stassun+, 2019)
 J/A+A/393/965  : Intrinsic freq. of slowly pulsating B stars (de Cat+, 2002)
 J/A+A/430/1143 : Stellar magnetic rotational phase curves (Bychkov+, 2005)
 J/other/A+ARV/18.67 : Accurate masses and radii of normal stars (Torres+, 2010)
 J/ApJS/215/15  : SMaSH+: observations and companion detection (Sana+, 2014)
 J/A+A/583/A115 : FORS1 catalogue of stellar magnetic fields (Bagnulo+, 2015)
 J/A+A/580/A27  : Asteroseismology of KIC 10526294 (Moravveji+, 2015)
 J/AJ/151/68    : Kepler Mission. VII. Eclipsing binaries in DR3 (Kirk+, 2016)
 J/A+A/598/A84  : OB-type spectroscopic binaries (Almeida+, 2017)
 J/other/Sci/359.69  : Massive stars in 30 Dor (Schneider+, 2018)

Byte-by-byte Description of file: table1.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label    Explanations
--------------------------------------------------------------------------------
   1-  9 I9     ---     TIC      [12359289/469906369] TESS Input Cat. identifier
  11- 11 A1     ---   f_TIC      [a] a: an LMC member
  13- 31 A19    ---     Name     Source name
  33- 46 A14    ---     SpType   SIMBAD spectral type
  48- 66 I19    ---     Gaia     ? Gaia DR2 identifier
  68- 72 F5.2   mag     GMag     [-4.6/1.8]? Absolute Gaia G band magnitude
  74- 78 F5.2   mag     BP-RP    [-0.4/3.2]? Gaia Blue-Red passbands color index
  80- 88 A9     ---     Nsp      Number(s) of high-resiolution spectra available
                                  in the ESO archive (2)
  90- 96 A7     ---     Inst     Instrument(s) (2)
  98-140 A43    ---     VarType  Variability type (3)
--------------------------------------------------------------------------------
Note (2): Observations on different instruments are separated by "," delimiters.
    Instrument code as follows:
    U = VLT/UVES;
    F = ESO/FEROS;
    X = VLT/X-SHOOTER;
    E = VLT/ESPRESSO.
Note (3): Variability type abbreviations:
         EB = eclipsing binary;
         EV = ellipsoidal variable;
        rot = Rotational modulation;
        SPB = Slowly Pulsating B star;
 {Beta} Cep = {Beta} Cephei star;
        SLF = stochastic low-frequency signal;
      instr = instrumental;
      const = constant;
       puls = pulsational signal not clearly identified in any
              of the previous categories;
{delta} Sct = {delta} Scuti star.
--------------------------------------------------------------------------------

History:
    From electronic version of the journal

================================================================================
(End)                    Prepared by [AAS], Emmanuelle Perret [CDS]  11-Aug-2020
