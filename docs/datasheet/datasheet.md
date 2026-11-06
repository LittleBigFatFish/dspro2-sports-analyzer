# Dataset Datasheet

**Course:** DSPRO2 HS26  
**Project:** Football Analyzer  
**Students:** Timo Tran, Johan Roth  
**Date:** 6 November 2026  
**Repository:** https://github.com/LittleBigFatFish/dspro2-sports-analyzer

We use six third-party football datasets, including access-controlled SoccerNet material. Five contain images and are used for player and ball detection; SkillCorner contains tracking data for the play analysis. Every field below covers all six.


## Dataset Overview

- **Official Name, Owner, Version & Date, Short Description, Reference:** see Table 1.

| **Official name** | **Owner** | **Version, date** | **Short description** | **Reference** |
| --- | --- | --- | --- | --- |
| Football Image Dataset | skaczylo (GitHub) | Commit `2642128`, 2026-06-23; downloaded 2026-10-20 | 612 football frames with ball, player and referee boxes in YOLO format. | github.com/skaczylo/Football-Image-Dataset |
| Soccana V1 (`Soccana_player_ball_detection_v1`) | Adit-jain (Hugging Face) | Revision `9e76891`, 2025-08-28; downloaded 2026-11-03 | Player, referee and ball boxes merged from Roboflow Universe sets; mostly image tiles. | huggingface.co/datasets/Adit-jain/Soccana_player_ball_detection_v1 |
| SoccerNet Tracking 2023 | SoccerNet team (University of Liège, KAUST) | Revision `3777f45`, 2026-09-06; downloaded 2026-11-03 | 30s broadcast clips with a tracked box per object, role, team and jersey. | A. Cioppa et al., CVPR Workshops 2022, arXiv:2204.06918 |
| SoccerNet Game State Reconstruction 2025 | SoccerNet team | Revision `ce84ee7`, 2024-10-22, labels v1.3; downloaded 2026-11-03 | Broadcast clips with boxes, pitch positions, teams and jerseys for a minimap. | V. Somers et al., CVPR Workshops 2024, arXiv:2404.11335 |
| SkillCorner Open Data | SkillCorner, with PySport | Commit `4340d27`, 2026-09-14; downloaded 2026-11-03 | Tracking of players and ball at 10fps and events for 20 A-League matches; no images. | github.com/SkillCorner/opendata |
| MultiSports (football part) | Multimedia Computing Group, Nanjing University | Revision `01600ce`, 2022-12-13; downloaded 2026-11-03 | Broadcast clips with action tubes: one box per frame on the player doing the action. | Y. Li et al., ICCV 2021, arXiv:2105.07404 |


## Motivation and Intended Use

- **Primary Purpose:** Football: a ready-to-use YOLO training set. Soccana: a starting point for a general football detector. SoccerNet Tracking and GSR: research benchmarks for tracking and minimap reconstruction. SkillCorner: open tracking data for analytics. MultiSports: a benchmark for action detection.
- **Intended Use:** Comparing pretrained YOLOv8n, YOLO12n and YOLO26n at 640 and 1280px on all image datasets, then fine-tuning (Football, Soccana, SoccerNet train). SoccerNet test and GSR valid are the main evaluation sets. The counts below describe the pretrained-model comparison; after fine-tuning, training images must not be reported as independent tests. Overlapping sources must be excluded from the held-out evaluation. Telling the two teams apart: SoccerNet and GSR give the team side of every player, SkillCorner the team of every tracked player. SkillCorner provides derived reference data for possession and passes, with missing observations and possible tracking errors.
- **Out-of-Scope Use:** Identifying people; team results on Football, Soccana or MultiSports (no team labels); claims beyond the leagues and games in the data; video-game frames reported as real footage; Soccana test or GSR test/train as separate results (they overlap); person precision or anything about the ball on MultiSports (only acting players are labelled); commercial use.

## Composition and Coverage

- **Data Types:** Unstructured: JPEG images and frames (Football, Soccana, SoccerNet, GSR), MP4 videos (MultiSports). Labels as YOLO text, COCO JSON, MOT text, JSON and a pickle. SkillCorner is structured: JSON, JSON lines, CSV.
- **Population:** Professional football from TV broadcasts. Football: club and national-team matches, among them Bundesliga, LaLiga, Saudi league, J-League, Copa Libertadores, FA Cup and Community Shield, three highlight clips and one video game. Soccana: seven Roboflow source groups, 71% tiles. SoccerNet and GSR: 9 games, probably the Swiss Super League (boards visible, not documented). MultiSports: YouTube broadcasts, among them the London 2012 Olympics and English club games. SkillCorner: A-League 2024/25.
- **Dataset Size:** Five image datasets with 440,231 images and frames and 2,273,795 labelled boxes, of which 11,153 frames are used in our tests (Table 2).
- **Known Gaps / Imbalances:** The ball is tiny: median shorter side 5.0-5.6px at 640px input on SoccerNet and Football. SoccerNet and GSR cover only 9 games. 43 of 100 Football val images are from a video game. Soccana is mostly tiles with 3,501 empty labels. MultiSports labels only acting players and never the ball. SkillCorner has no tracking in 25.7% of in-play frames. Rain, snow, amateur and youth games are not established as covered by our audit.

| **Dataset** | **Images / frames** | **Boxes** | **Ball boxes** | **Used in tests** |
| --- | --- | --- | --- | --- |
| Football Image Dataset | 612 | 10,136 | 500 | 612 |
| Soccana V1 | 11,673 | 64,648 | 3,600 | 500 |
| SoccerNet Tracking 2023 (106 clips) | 79,500 | 1,297,549 | 75,060 | 3,180 |
| SoccerNet GSR 2025, valid (58 clips) | 43,500 | 748,666 | 40,900 | 1,740 |
| MultiSports, football (534 clips) | 304,946 | 152,796 | - | 5,121 |
| **Image datasets** | **440,231** | **2,273,795** | **120,060** | **11,153** |
| SkillCorner (20 matches, tracking) | 1,244,814 | 94,361 events | ball in 74% | - |


## Sources and Collection

- **Provenance:** Third-party broadcast footage and video-game imagery, distributed through public or controlled-access sources. 48 of the 100 Football val images are also in Soccana train, so both share an upstream source. GSR test and train are byte-identical to SoccerNet Tracking. Our overlap check found no shared frames between MultiSports and Football or Soccana.
- **Collection Method:** Football: frames cut from videos. Soccana: source sets sliced into tiles and capped per source. SoccerNet and GSR: 30s clips at 25fps, 1920 x 1080. MultiSports: YouTube videos cut into clips, 1280 x 720 at 25fps. SkillCorner: computer vision on broadcast video at 10fps, positions outside the view extrapolated.
- **Jurisdictions:** Football: Germany, Spain, England, Saudi Arabia, Japan, South Korea, South America and international matches. SoccerNet and GSR: probably Switzerland. SkillCorner: Australia and New Zealand. Soccana and MultiSports: not documented.

## Labels and Targets

- **Target Definition:** Boxes for ball, player (goalkeepers included) and referee, mapped to one shared scheme (0 ball, 1 player, 2 referee). SoccerNet "other" (staff) is ignored. Every MultiSports box counts as a player. For COCO models, player and referee count as "person", ball as "sports ball". Team labels: team side per player in SoccerNet and GSR, team per player in SkillCorner. SkillCorner: possessions and passes.
- **Labeling Process:** Football: manually annotated. Soccana: taken over from the source sets and merged. SoccerNet and GSR: by the authors (papers). MultiSports: professional athletes marked actions, crowd workers corrected the boxes. SkillCorner: events derived from the tracking data.
- **Quality Control:** Our audits of every split and 20 overlays per class checked by eye (Soccana, SoccerNet, GSR; 20 val overlays for MultiSports). GSR: players 20 of 20 and balls 18 of 20 correct, referees correct but 4 of 20 hard to see in blurred frames. Football spot-check of 34 images (5 per source, 4 from the smallest): Among 580 inspected boxes, no wrong class was found; one goalkeeper annotation was missing and three cases were unclear.

## Quality Assessment (Data Audit)

- **Preprocessing Steps:** None on the raw data. Converted copies in YOLO format, lists of excluded images and the shared class scheme are kept separately. For inference, images are only resized to 640 or 1280px.
- **Initial Audit Findings:** Football: 0 structural audit errors, 31 warnings and two separately identified label problems. Soccana: the test split leaks (135 test images have a twin in train), 81 boxes outside the image; 209 images excluded by the audit and 71 by the overlap check, and the group overlapping Football dropped. SoccerNet, GSR, MultiSports: 0 structural audit errors. SkillCorner: no missing frames; 1,534 player and 1,567 ball positions lie more than 5m beyond the touchline or goal line. We use a 5m margin so that normal play near the lines is not counted.
- **Quality Dimensions:** Completeness: every image has a label file, which may be empty; GSR pitch positions and SkillCorner positions are partly missing. Accuracy: labels mostly correct in our overlays; the Football spot-check found a missing annotation and unclear cases. Consistency: different class IDs and formats, unified in our converted copies. Timeliness: known match dates from 2012 to 2025; every version pinned by commit or revision. Zero-shot at 1280px: person AP50 .83-.86 on SoccerNet and .95-.97 on real Football val, ball recall only .27-.32 on SoccerNet.

## Legal, Ethical, and Governance

- **Sensitive Data:** Our source footage shows professional players, officials and spectators; faces are not blurred. We analyse players and teams without identifying individuals or linking tracks to names. SkillCorner includes names and birth dates, but we use player IDs only. We keep the datasets local and do not redistribute them.
- **Compliance:** We conduct this project exclusively for educational purposes at Lucerne University of Applied Sciences and Arts (HSLU). Our model detects players and the ball for football analysis, without facial recognition or individual identification. We treat the current pre-deployment research and development work under the EU AI Act research exclusion (Art. 2(8)); real-world testing and later deployment require their own assessment. We handle identifiable source footage and player metadata under applicable data-protection requirements (GDPR / Swiss FADP), using only the data needed for our analysis [gdpr, aiact, fadp].
- **Consent & Licensing:** Our SoccerNet NDA permits both team members, Timo Tran and Johan Roth, to use the data for this HSLU project, provided we do not redistribute it. We cite the original sources. SoccerNet allows cited examples in research papers and presentations [sndata, snfaq]. Dataset-specific licences are summarised in Table 3.

| **Dataset** | **Licence** | **What the licence says** | **Consent and our use** |
| --- | --- | --- | --- |
| Football Image Dataset | CC BY 4.0 (licence file of the dataset) | Use, share and adapt for any purpose if the author is credited. Covers the annotations; the frames are third-party broadcasts. | Consent of the people shown not mentioned. Coursework only, images not shared, author credited. |
| Soccana V1 | No aggregate licence declared | Upstream licences apply. Five linked source pages declare CC BY 4.0; three source licences could not be verified. The card lists eight sources, while our local audit identifies seven groups. A complete source-to-file licence mapping is not established [soccana, rfAS, rfRF, rfSAC, rfVA, rfVAA]. | Individual consent is not stated in the dataset card. No redistribution; training-use permission remains unresolved for unverified sources. |
| SoccerNet Tracking 2023 | Research terms; no explicit licence on public card | SoccerNet permits model training for non-commercial research. Broadcast copyright remains with the rights holders. Cite the dataset; do not redistribute protected material [sndata, snfaq, trackingcard]. | Our NDA permits both of us to use the data for this HSLU educational project. We do not redistribute it. Individual consent is not stated in the public FAQ. |
| SoccerNet GSR 2025 | GPL-3.0 tag on dataset card | The card declares GPL-3.0; its scope for third-party imagery is not clarified. Apply SoccerNet research restrictions to the footage; the tag is not treated as unrestricted broadcast rights [snfaq, gsrCard]. | We use the data for our HSLU educational project under our SoccerNet NDA, without redistribution. We cite the paper. Individual consent is not stated in the dataset card. |
| SkillCorner Open Data | MIT in repository | Keep the copyright and permission notice when redistributing covered material. Credit SkillCorner. Repository licensing is distinct from personal-data permissions [skillcorner]. | Contains names and birth dates; we use IDs only. Individual consent is not stated in the repository. IDs may remain linkable to players. |
| MultiSports | CC BY-NC 4.0 | Attribution and non-commercial use required; indicate changes. The access page requires acceptance of the licence. Third-party footage may carry additional rights [multisportscard]. | Access acceptance is recorded in our existing audit. Individual consent is not stated in the dataset card. Coursework only, paper cited. |


## Dataset Structure and Schema

- **File Formats:** JPEG, YOLO `.txt`, COCO JSON, MOT `.txt` and `.ini`, GSR JSON, MP4 with a pickle, SkillCorner JSON, JSON lines and CSV.
- **Directory Layout:** Raw inputs include `data/raw/Football-Image-Dataset/` and `data/raw/soccana_v1/`. Converted evaluation data is under `data/processed/soccernet/`, `data/processed/gsr/` and `data/processed/sportsaction/`; audit outputs are kept separately under `outputs/`. Raw downloads are not edited.
- **Schema Summary:** Every image dataset is converted to YOLO: one label file per image, one line per box: `class x_center y_center width height`. SkillCorner: one tracking line per frame; events link to frames and player IDs.
- **Units & Encodings:** YOLO coordinates normalised to [0,1]; COCO, MOT and GSR boxes in pixels; GSR and SkillCorner positions in metres; UTF-8 text.

## Credits and References

We credit the original dataset providers and papers [soccana, tracking, multisports, football, skillcorner, gsr]. YOLO models and software are provided by Ultralytics under their published licensing terms [ultralytics]. Source and terms pages were checked on 6 November 2026. The data counts and findings above come from our project audit.
\footnotesize

- **[football]** skaczylo. [Football Image Dataset](https://github.com/skaczylo/Football-Image-Dataset). Accessed 2026-11-06.
- **[soccana]** Adit-jain. [Soccana player ball detection v1: Dataset Card](https://huggingface.co/datasets/Adit-jain/Soccana_player_ball_detection_v1). Accessed 2026-11-06.
- **[tracking]** Cioppa, Anthony et al.. [SoccerNet-Tracking: Multiple Object Tracking Dataset and Benchmark in Soccer Videos](https://arxiv.org/abs/2204.06918). Accessed 2026-11-06.
- **[gsr]** Somers, Vladimir et al.. [SoccerNet Game State Reconstruction: End-to-End Athlete Tracking and Identification on a Minimap](https://arxiv.org/abs/2404.11335). Accessed 2026-11-06.
- **[multisports]** Li, Yixuan et al.. [MultiSports: A Multi-Person Video Dataset of Spatio-Temporally Localized Sports Actions](https://arxiv.org/abs/2105.07404). Accessed 2026-11-06.
- **[skillcorner]** SkillCorner. [Open Data Repository and MIT License](https://github.com/SkillCorner/opendata). Accessed 2026-11-06.
- **[snfaq]** SoccerNet. [Frequently Asked Questions](https://www.soccer-net.org/faq). Accessed 2026-11-06.
- **[sndata]** SoccerNet. [Data Access and Download Instructions](https://www.soccer-net.org/data). Accessed 2026-11-06.
- **[trackingcard]** SoccerNet. [SN-Tracking-2023 Dataset Repository](https://huggingface.co/datasets/SoccerNet/SN-Tracking-2023). Accessed 2026-11-06.
- **[gsrCard]** SoccerNet. [SN-GSR-2025 Dataset Card](https://huggingface.co/datasets/SoccerNet/SN-GSR-2025). Accessed 2026-11-06.
- **[multisportscard]** Multimedia Computing Group, Nanjing University. [MultiSports Dataset Card and Access Conditions](https://huggingface.co/datasets/MCG-NJU/SportsAction). Accessed 2026-11-06.
- **[ultralytics]** Ultralytics. [Licensing](https://www.ultralytics.com/license). Accessed 2026-11-06.
- **[aiact]** European Union. [Regulation (EU) 2024/1689: Artificial Intelligence Act](https://eur-lex.europa.eu/eli/reg/2024/1689/oj). Accessed 2026-11-06.
- **[gdpr]** European Union. [Regulation (EU) 2016/679: General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj). Accessed 2026-11-06.
- **[fadp]** Swiss Confederation. [Federal Act on Data Protection](https://www.fedlex.admin.ch/eli/cc/2022/491/en). Accessed 2026-11-06.
- **[rfSAC]** SAC. [Roboflow Universe source: soccer-player-tracker](https://universe.roboflow.com/sac-wjhag/soccer-player-tracker). Accessed 2026-11-06.
- **[rfVAA]** VAA. [Roboflow Universe source: va_project-mp2xn](https://universe.roboflow.com/vaa/va_project-mp2xn). Accessed 2026-11-06.
- **[rfVA]** VA. [Roboflow Universe source: football-eitpt](https://universe.roboflow.com/va-sah7v/football-eitpt). Accessed 2026-11-06.
- **[rfAS]** Augmented Startups. [Roboflow Universe source: football-player-detection-kucab](https://universe.roboflow.com/augmented-startups/football-player-detection-kucab). Accessed 2026-11-06.
- **[rfRF]** Roboflow. [Roboflow Universe source: football-players-detection-3zvbc](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc). Accessed 2026-11-06.
