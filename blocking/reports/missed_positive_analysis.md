# Missed positive analysis

| Failure category | Missed pairs |
| --- | --- |
| high multiplicity | 3707 |
| no shared normalized name token | 1199 |
| different script/non-ASCII transition | 1191 |
| partial lexical corruption/other | 771 |
| missing target address | 728 |
| short/generic query name | 711 |
| name word order | 231 |
| no shared normalized address token | 3 |

## Representative examples

- **S1-915500283 → S3-773888102**: high multiplicity. Query `Todd Assets, LLC` / `204 Jay Street, Rockford, OH`; target `Brixveo F/K/A Todd Assets, LLC` / `204 Jay Street, Rockford, OH`.
- **S1-915500283 → S3-669375114**: high multiplicity. Query `Todd Assets, LLC` / `204 Jay Street, Rockford, OH`; target `Todd Aimstes, LLC` / `Rockford, Ohio, 204 Jay Street`.
- **S1-915500283 → S3-129911140**: high multiplicity. Query `Todd Assets, LLC` / `204 Jay Street, Rockford, OH`; target `Todd Assets, Llc` / `204 Jay St, Rockford, Ohio`.
- **S1-915500283 → S2-700990164**: different script/non-ASCII transition, high multiplicity. Query `Todd Assets, LLC` / `204 Jay Street, Rockford, OH`; target `Todd Ássets, LLC` / `JAY STREET, ROCKFORD, OH`.
- **S1-250245862 → S2-198183335**: partial lexical corruption/other. Query `Rev It Up Nails LLC` / `24 Elm Street, Greenwich, CT`; target `REV REV IT-UP` / `24- ELM STREET, GREENWICH, CT`.
- **S1-250245862 → S3-578359010**: partial lexical corruption/other. Query `Rev It Up Nails LLC` / `24 Elm Street, Greenwich, CT`; target `Rev It Up` / `Connecticut, Greenwich, Elm Street`.
- **S1-616650728 → S3-472557282**: short/generic query name, high multiplicity. Query `JJ Viking` / `16 Normandy Avenue, Cambridge, MA`; target `JJ Vcikg` / `16 Normandy Ave, Cambridge, Massachusetts`.
- **S1-616650728 → S2-431319560**: short/generic query name, high multiplicity. Query `JJ Viking` / `16 Normandy Avenue, Cambridge, MA`; target `JJ  Viking` / `NORMANDY AVE, CAMBRIDGE, MA`.
- **S1-616650728 → S3-146261167**: short/generic query name, high multiplicity. Query `JJ Viking` / `16 Normandy Avenue, Cambridge, MA`; target `JJ  Viking` / `Massachusetts, Cambridge, 16 Normandy Ave`.
- **S1-616650728 → S3-185042572**: short/generic query name, high multiplicity. Query `JJ Viking` / `16 Normandy Avenue, Cambridge, MA`; target `JJ  Viking` / `Massachusetts, Cambridge, 16 Normandy Avenue`.
- **S1-203233410 → S2-750914769**: different script/non-ASCII transition, no shared normalized name token. Query `Red Infra Private Limited` / `No 93Eswaran Kovil, St Anaikatticherry, Poonamallee, Tiruvallur, Tamil Nadu`; target `ரெட் இன்ஃப்ரா பிரைவேட் லிமிடெட்` / `NO G-93ESWARAN KOVIL, CHENNAI, POONAMALLEE, தமிழ்நாடு`.
- **S1-203233410 → S2-271772318**: different script/non-ASCII transition, no shared normalized name token. Query `Red Infra Private Limited` / `No 93Eswaran Kovil, St Anaikatticherry, Poonamallee, Tiruvallur, Tamil Nadu`; target `ரெட் இன்ஃப்ரா பிரைவேட் லிமிடெட்` / `DOOR NO G-93ESWARAN KOVIL, CHENNAI, TIRUVALLUR, Tamil Nadu`.
- **S1-803732599 → S2-979565211**: high multiplicity. Query `Eye Patriot Physicians LP` / `45 Rhodes Cove Road, Sylva, NC`; target `Eye Patriot Physicians LLC` / `##45 RHODES COVE RD, SLYVA, NC`.
- **S1-803732599 → S2-280232228**: high multiplicity. Query `Eye Patriot Physicians LP` / `45 Rhodes Cove Road, Sylva, NC`; target `EYE PATRIOT PHYSICIANS PHYSICIANS LLC` / `##45 RHODES COVE RD, SLYVA, NC`.
- **S1-916121746 → S3-839355975**: no shared normalized name token, high multiplicity. Query `Parasram Krishan Limited` / `1325/1/K/4, 1St Floor, Unit No F 5, Sanghavi Height, Kolhapur, Maharashtra`; target `Rizaxylo` / `1325/1/K/4, Kolhapur, MH`.
- **S1-814583640 → S3-432294658**: different script/non-ASCII transition. Query `Pioneer Foundation Private Limited` / `H.No 10/A, Flat No 313, B Block Ameena Complex, Uppal, Hyderabad, Telangana`; target `పయనీర్ Foundation Private Limited` / `H.no 10/A, Flat No 313, B Block Ameena Complex, Hyderabad, TG`.
- **S1-814583640 → S2-903941192**: different script/non-ASCII transition, no shared normalized name token. Query `Pioneer Foundation Private Limited` / `H.No 10/A, Flat No 313, B Block Ameena Complex, Uppal, Hyderabad, Telangana`; target `పయనీర్ ఫౌండేషన్ ప్రైవేట్ లిమిటెడ్` / `H.NO 10/A, FLAT NO 313, B BLOCK AMEENA COMPLEX, K.V.RANGAREDDY, HYDERABAD, Andhra Pradesh`.
- **S1-201214059 → S3-318142856**: no shared normalized name token, high multiplicity. Query `Classic Agro Pvt Ltd` / `H. No. 8-2-351/N/3, 5Th Floor, Nishant House, Road No. 3, Khairatabad, Hyderabad, Telangana`; target `Smt classicagro.com` / `##8-2-351/N/3, Hyderabad, తెలంగాణ`.
- **S1-898548516 → S3-996613494**: high multiplicity. Query `Jay City Projects Private Limited` / `11-6-211/201, Goldenheight, Pochammabagh, Hyderabad, Telangana`; target `JAY CITY PROJECTS PRIVATE LIMITED` / `11-6-211/201, Hyderabad, తెలంగాణ`.
- **S1-898548516 → S3-210715605**: different script/non-ASCII transition, high multiplicity. Query `Jay City Projects Private Limited` / `11-6-211/201, Goldenheight, Pochammabagh, Hyderabad, Telangana`; target `Jay Cíty Projects Private` / `11-6-211/201, Hyderabad, TG`.
- **S1-941294459 → S3-732958244**: partial lexical corruption/other. Query `Bt Realty Private Limited` / `Hd-295, Sy.130P&115/1P, Nanakramguda, Hyderabad, Seri Lingampally, K.V.Rangareddy, Telangana`; target `Smt Bt  Realty Private` / `Hn 312 Hd-295, Seri Lingampally, K.v.rangareddy, తెలంగాణ`.
- **S1-941294459 → S2-673951933**: partial lexical corruption/other. Query `Bt Realty Private Limited` / `Hd-295, Sy.130P&115/1P, Nanakramguda, Hyderabad, Seri Lingampally, K.V.Rangareddy, Telangana`; target `Shri Bt Realty  Limited Center` / `HD-95, SY.130P&115/1P, NANAKRAMGUDA, HYDERABAD, SERI LINGAMPALLY, Telangana`.
- **S1-733968062 → S3-703219647**: missing target address. Query `Capital Foundation Inc` / `82 Blythe Ranch Road, Trinity, TX`; target `Capital-Foundariton Inc` / ``.
- **S1-788562064 → S3-389462125**: partial lexical corruption/other. Query `Behavioral Health United Care` / `848 Euclid Avenue, Village Of Wild Rose, WI`; target `Behavioral Health United Care  LLC` / `848 Euclid Avenue, Wisconsin, Village Of Wild Rose`.
- **S1-788562064 → S2-159581122**: partial lexical corruption/other. Query `Behavioral Health United Care` / `848 Euclid Avenue, Village Of Wild Rose, WI`; target `Behavioral Healt United Care` / `EUCLID AVE, VILLAGE OF WILD ROSE, WI`.
- **S1-788562064 → S2-937940237**: partial lexical corruption/other. Query `Behavioral Health United Care` / `848 Euclid Avenue, Village Of Wild Rose, WI`; target `Behavioral Health` / `EUCLID AVENUE, VILLAGE OF WILD ROSE, WI`.
- **S1-485212696 → S2-372981353**: name word order, high multiplicity. Query `Jany W. Joyce, P.A., PC` / `14052 30th Avenue, Seattle, WA`; target `Joyce, W. Jany P.A., PC` / `30ST AVENUE, SEATTLE, WA`.
- **S1-658298667 → S3-169839257**: high multiplicity. Query `Foot & Ankle Great Center` / `89 Mill Rock Road, Old Saybrook, CT`; target `Foot & Ankle Great Cfmftr` / `269- Mill Rock Road, Old Saybrook, Connecticut`.
- **S1-658298667 → S2-331294170**: high multiplicity. Query `Foot & Ankle Great Center` / `89 Mill Rock Road, Old Saybrook, CT`; target `Foot & Ankle Great` / `N/A, 269 MILL ROCK RD, OLD SAYBROOK, CT`.
- **S1-644908633 → S2-527964201**: missing target address, different script/non-ASCII transition. Query `Consultancy Aar Impex Ltd` / `Kagal, Maharashtra, H No.544/B, Sarpiraji Road A/P Murgud, Kolhapur`; target `CONSULTANCY ÁAR IMPEX LTD` / ``.
