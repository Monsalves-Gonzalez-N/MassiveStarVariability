J/MNRAS/485/3457      Rotational modulation in TESS B stars      (Balona+, 2019)
================================================================================
Rotational modulation in TESS B stars.
    Balona L.A., Handler G., Chowdhury S., Ozuyar D., Engelbrecht C.A.,
    Mirouh G.M., Wade G.A., David-Uraz A., Cantiello M.
   <Mon. Not. R. Astron. Soc., 485, 3457-3469 (2019)>
   =2019MNRAS.485.3457B    (SIMBAD/NED BibCode)
================================================================================
ADC_Keywords: Stars, early-type ; Stars, B-type ; Stars, variable ;
              Rotational velocities ; Effective temperatures ; Optical
Keywords: stars: early-type - stars: oscillations - stars: rotation

Abstract:
    Light curves and periodograms of 160 B stars observed by the
    Transiting Exoplanet Survey Satellite (TESS) space mission and 29
    main-sequence B stars from Kepler and K2 were used to classify the
    variability type. There are 114 main-sequence B stars in the TESS
    sample, of which 45 are classified as possible rotational variables.
    This confirms previous findings that a large fraction (about 40 per
    cent) of A and B stars may exhibit rotational modulation. Gaia DR2
    parallaxes were used to estimate luminosities, from which the radii
    and equatorial rotational velocities can be deduced. It is shown that
    observed values of the projected rotational velocities are lower than
    the estimated equatorial velocities for nearly all the stars, as they
    should be if rotation is the cause of the light variation. We conclude
    that a large fraction of main-sequence B stars appear to contain
    surface features which cannot likely be attributed to abundance
    patches.

Description:
    TESS observes the sky in sectors measuring 24{deg}x96{deg} that extend
    from near the ecliptic equator to beyond the ecliptic pole. Each
    sector is observed for two orbits of the satellite around the Earth,
    or about 27d.

    The data analysed in this paper are from the light curves of the first
    release (Sectors 1 and 2). Most of the stars discussed here are at
    mid-latitudes and have been observed for a time span of about 55d. A
    description of how these data products were generated is found in
    Jenkins et al. (2016SPIE.9913E..3EJ).

    The TESS input catalogue (Stassun et al. 2018AJ....156..102S, Cat.
    J/AJ/156/102) lists stellar parameters for stars observed by TESS. The
    effective temperatures, Teff, for stars without spectroscopic
    determinations were obtained from near-infrared photometry, which is
    not reliable for B stars, particularly since reddening is important in
    many cases. The stars observed by TESS were matched with the SIMBAD
    astronomical data base (Wenger et al. 2000A&AS..143....9W) and only
    those known to be B stars were selected, giving a total of 160 stars
    with spectral types earlier than A0 (see Table 1).

    In addition to the TESS stars, we have examined the light curves of
    Kepler (Kepler Mission Team, 2009yCat.5133....0K, Cat. V/133) and K2
    (Huber et al. 2017yCat.4034....0H, Cat. IV/34) data for possible
    rotational modulation. The Kepler data have a time span of nearly 4yr
    which results in a very low periodogram noise level. The K2 data have
    a time span of around 80d. Table 3 lists the measurements and stellar
    parameters obtained for Kepler and K2 light curves.

File Summary:
--------------------------------------------------------------------------------
 FileName      Lrecl  Records   Explanations
--------------------------------------------------------------------------------
ReadMe            80        .   This file
table1.dat       109      160   List of B stars observed by TESS
table3.dat       108       29   Additional rotational variables identified from
                                the light curves of the Kepler and K2 missions
--------------------------------------------------------------------------------

See also:
    J/AJ/156/102 : The TESS Input Catalog and Candidate Target List
                                                   (Stassun+, 2018)
           V/133 : Kepler Input Catalog (Kepler Mission Team, 2009)
           IV/34 : K2 Ecliptic Plane Input Catalog (EPIC) (Huber+, 2017)
           I/345 : Gaia DR2 (Gaia Collaboration, 2018)
           IV/39 : TESS Input Catalog version 8.2 (TIC v8.2) (Paegert+, 2021)

Byte-by-byte Description of file: table1.dat
--------------------------------------------------------------------------------
   Bytes Format Units    Label     Explanations
--------------------------------------------------------------------------------
   1-  9  I9    ---      Name      TIC star name (NNNNNNNNN)
  11- 25  A15   ---      OName     Other name
  27- 40  A14   ---      VarType   Assigned variability type (GCVS type in
                                    parenthesis from Samus et al.
                                    2017ARep...61...80S, Cat. B/gcvs)
  42- 47  F6.3  mag      Vmag      Apparent V magnitude
  49- 53  F5.3  d-1      nurot     ? Presumed rotational frequency (in cycles/d)
  55- 59  I5    ppm      Arot      ? Amplitude of the periodogram peak
  61- 63  I3    ---      S/N       ? Signal to noise ratio of the periodogram
                                    peak
      65  I1    ---      NH        ? Number of visible harmonics
  67- 69  I3    km/s     ve        ? Derived equatorial rotational velocity (G1)
  71- 73  I3    km/s     vsini     ? Projected rotational velocity from the
                                    literature
  75- 79  I5    K        Teff      ? Adopted effective temperature
  81- 82  I2    ---    r_Teff      ? Reference for Teff (G2)
  84- 88  F5.2  [Lsun]   logLstar  ? Stellar luminosity obtained from the Gaia
                                    parallax (Gaia Collaboration
                                    2018A&A...616A...1G, Cat. I/345)
  90-109  A20   ---      SpType    Spectral type ((Be) indicates classical Be
                                    star)
--------------------------------------------------------------------------------

Byte-by-byte Description of file: table3.dat
--------------------------------------------------------------------------------
   Bytes Format Units    Label     Explanations
--------------------------------------------------------------------------------
   1- 14  A14   ---      Name      Star name from KIC/EPIC
  16- 27  A12   ---      OName     Other name
  29- 39  A11   ---      VarType   Assigned variability type (GCVS type in
                                    parenthesis from Samus et al.
                                    2017ARep...61...80S, Cat. B/gcvs)
  41- 46  F6.3  mag      Vmag      Apparent V magnitude
  48- 52  F5.3  d-1      nurot     Presumed rotational frequency (in cycles/d)
  54- 58  I5    ppm      Arot      Amplitude of the periodogram peak
  60- 61  I2    ---      S/N       Signal to noise ratio of the periodogram peak
      63  I1    ---      NH        Number of visible harmonics
  65- 67  I3    km/s     ve        Derived equatorial rotational velocity (G1)
  69- 71  I3    km/s     vsini     Projected rotational velocity from the
                                    literature
  73- 77  I5    K        Teff      Adopted effective temperature
  79- 80  I2    ---    r_Teff      Reference for Teff (G2)
  82- 85  F4.2  [Lsun]   logLstar  Stellar luminosity obtained from the Gaia
                                    parallax (Gaia Collaboration
                                    2018A&A...616A...1G, Cat. I/345)
  87-108  A22   ---      SpType    Spectral type ((Be) indicates classical Be
                                    star)
--------------------------------------------------------------------------------

Global Notes:
Note (G1): The equatorial rotational velocity is given by
           ve=50.74nurot(R/R_{sun}_)
Note (G2): References as follows:
   1 = Pecaut & Mamajek (2013ApJS..208....9P, Cat. J/ApJS/208/9) Spectral type
   2 = Urbaneja et al. (2017AJ....154..102U, Cat. J/AJ/154/102) Spectroscopy
   3 = Gianninas, Bergeron & Ruiz (2011ApJ...743..138G, Cat. J/ApJ/743/138)
       Spectroscopy
   4 = Paunzen, Schnell & Maitzen (2005A&A...444..941P, Cat. J/A+A/444/941)
       Narrow-band photometry
   5 = Wright et al. (2003AJ....125..359W, Cat. III/231) Spectral type
   6 = Balona (1994MNRAS.268..119B) Stromgren
   7 = Chandler, McDonald & Kane (2016AJ....151...59C, Cat. J/AJ/151/59)
       SED fitting
   8 = Gullikson, Kraus & Dodson-Robinson (2016AJ....152...40G,
       Cat. J/AJ/152/40 ) Spectroscopy
   9 = Sanchez-Blazquez et al. (2006MNRAS.371..703S, Cat. J/MNRAS/371/703)
       Spectroscopy
  10 = McDonald, Zijlstra & Watson (2017MNRAS.471..770M, Cat. J/MNRAS/471/770)
       SED fitting
  11 = Geier et al. (2017A&A...600A..50G, Cat. J/A+A/600/A50) Stectral type/SED
  12 = Oey & Massey (1995ApJ...452..210O, Cat. J/ApJ/452/210) Sp.Type and UBV
  13 = Paunzen et al. (2013MNRAS.429..119P, Cat. J/MNRAS/429/119)
       UBV/Geneva/Stromgren
  14 = David & Hillenbrand (2015ApJ...804..146D, Cat. J/ApJ/804/146) Stromgren
  15 = De Cat & Aerts (2002A&A...393..965D, Cat. J/A+A/393/965) Geneva
  16 = Silva & Napiwotzki (2011MNRAS.411.2596S, Cat. J/MNRAS/411/2596) Stromgren
  17 = Zorec et al. (2009A&A...501..297Z, Cat. J/A+A/501/297) BCD method
  18 = Silaj & Landstreet (2014A&A...566A.132S, Cat. J/A+A/566/A132)
       Geneva/Stromgren
  19 = Soubiran et al. (2016A&A...591A.118S, Cat. B/pastel) Spectral type
  20 = Massey (2002ApJS..141...81M, Cat. II/236) UBVR photometry
  21 = Niemczura et al. (2015MNRAS.450.2764N, Cat. J/MNRAS/450/2764)
       Spectroscopy
  22 = Huber et al. (2016ApJS..224....2H, Cat. J/ApJS/224/2) Spectroscopy
  23 = Balona et al. (2011MNRAS.413.2403B) Spectroscopy
  24 = Tkachenko et al. (2013MNRAS.431.3685T) Spectroscopy
--------------------------------------------------------------------------------

History:
    From electronic version of the journal

================================================================================
(End)                                           Ana Fiallos [CDS]    23-Sep-2022
