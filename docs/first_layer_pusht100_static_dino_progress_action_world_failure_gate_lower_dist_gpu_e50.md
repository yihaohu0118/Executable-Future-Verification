# DINO-Failure-Aware UMM World-Model Gate

## Bottom Line

This trains a fold-wise binary gate that decides whether a case should stay with the static DINO selector or be overridden by a UMM world-model selector. The gate uses only non-oracle confidence, planner, and action-conditioned rollout features at test time.

## Result

| Objective | Mean dist | Better | Oracle match | Pair acc | Expert acc | Gate acc | Overrides |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lower_dist | 184.2875 | 52/100 | 36/100 | 0.5905 | 0.6366 | 0.6400 | 38/100 |

## Case Decisions

| Fold | Case | Pred | Target | DINO dist | UMM dist | Oracle dist | Rescue | Loss |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | pusht_episode_084_step_000 | DINO | DINO | 133.5991 | 133.5991 | 128.9333 | 0 | 0 |
| 0 | pusht_episode_086_step_000 | DINO | DINO | 111.4477 | 137.8223 | 70.1992 | 0 | 0 |
| 0 | pusht_episode_065_step_000 | UMM | DINO | 166.1039 | 245.3156 | 166.1039 | 0 | 1 |
| 0 | pusht_episode_045_step_000 | DINO | DINO | 193.2041 | 193.2041 | 150.8250 | 0 | 0 |
| 0 | pusht_episode_061_step_000 | DINO | DINO | 70.9816 | 70.9816 | 70.9816 | 0 | 0 |
| 0 | pusht_episode_010_step_000 | UMM | DINO | 244.2671 | 317.2537 | 244.2671 | 0 | 1 |
| 0 | pusht_episode_093_step_000 | UMM | UMM | 164.1270 | 148.9881 | 121.7164 | 0 | 0 |
| 0 | pusht_episode_087_step_000 | DINO | DINO | 240.2391 | 240.2391 | 240.2391 | 0 | 0 |
| 0 | pusht_episode_059_step_000 | DINO | DINO | 252.5808 | 252.5808 | 252.5808 | 0 | 0 |
| 0 | pusht_episode_098_step_000 | UMM | DINO | 222.5428 | 244.0301 | 208.4497 | 0 | 0 |
| 0 | pusht_episode_088_step_000 | UMM | DINO | 115.2581 | 194.6293 | 115.2581 | 0 | 1 |
| 0 | pusht_episode_076_step_000 | UMM | UMM | 125.9020 | 86.1742 | 86.1742 | 1 | 0 |
| 0 | pusht_episode_003_step_000 | DINO | UMM | 232.6478 | 195.0729 | 133.1285 | 0 | 0 |
| 0 | pusht_episode_095_step_000 | UMM | UMM | 191.4055 | 119.6308 | 119.6308 | 1 | 0 |
| 0 | pusht_episode_089_step_000 | UMM | DINO | 119.1202 | 132.6987 | 119.1202 | 0 | 1 |
| 1 | pusht_episode_006_step_000 | DINO | DINO | 91.2389 | 116.6393 | 91.2389 | 0 | 1 |
| 1 | pusht_episode_030_step_000 | UMM | UMM | 162.2721 | 137.4281 | 104.5054 | 0 | 0 |
| 1 | pusht_episode_047_step_000 | DINO | DINO | 116.5279 | 162.4880 | 116.5279 | 0 | 1 |
| 1 | pusht_episode_079_step_000 | DINO | DINO | 168.7012 | 168.7012 | 168.7012 | 0 | 0 |
| 1 | pusht_episode_071_step_000 | DINO | DINO | 184.2835 | 184.2835 | 155.8825 | 0 | 0 |
| 1 | pusht_episode_011_step_000 | DINO | DINO | 328.7305 | 328.7305 | 291.4912 | 0 | 0 |
| 1 | pusht_episode_021_step_000 | UMM | DINO | 218.4669 | 250.8364 | 218.4669 | 0 | 1 |
| 1 | pusht_episode_083_step_000 | DINO | DINO | 124.9349 | 124.9349 | 102.4738 | 0 | 0 |
| 1 | pusht_episode_056_step_000 | UMM | UMM | 157.3903 | 135.4677 | 135.4677 | 1 | 0 |
| 1 | pusht_episode_016_step_000 | UMM | UMM | 208.0284 | 168.9526 | 168.9526 | 1 | 0 |
| 1 | pusht_episode_004_step_000 | UMM | UMM | 149.0753 | 116.6238 | 114.7030 | 0 | 0 |
| 1 | pusht_episode_080_step_000 | DINO | UMM | 173.7162 | 164.7224 | 162.0198 | 0 | 0 |
| 1 | pusht_episode_028_step_000 | DINO | DINO | 194.9867 | 194.9867 | 194.9867 | 0 | 0 |
| 1 | pusht_episode_039_step_000 | UMM | UMM | 218.6881 | 155.4751 | 144.0669 | 0 | 0 |
| 1 | pusht_episode_074_step_000 | DINO | UMM | 266.2802 | 251.5667 | 251.5667 | 1 | 0 |
| 2 | pusht_episode_012_step_000 | DINO | DINO | 138.7300 | 152.1405 | 122.8260 | 0 | 0 |
| 2 | pusht_episode_054_step_000 | DINO | UMM | 291.4391 | 200.3871 | 185.0785 | 0 | 0 |
| 2 | pusht_episode_077_step_000 | DINO | DINO | 173.0203 | 173.0203 | 173.0203 | 0 | 0 |
| 2 | pusht_episode_032_step_000 | DINO | DINO | 190.2089 | 190.2089 | 190.2089 | 0 | 0 |
| 2 | pusht_episode_052_step_000 | UMM | UMM | 275.1399 | 152.5085 | 152.5085 | 1 | 0 |
| 2 | pusht_episode_040_step_000 | UMM | UMM | 184.8285 | 175.5304 | 153.3040 | 0 | 0 |
| 2 | pusht_episode_064_step_000 | DINO | DINO | 109.7803 | 109.7803 | 109.7803 | 0 | 0 |
| 2 | pusht_episode_014_step_000 | DINO | DINO | 173.0271 | 203.7301 | 173.0271 | 0 | 1 |
| 2 | pusht_episode_044_step_000 | UMM | DINO | 232.9158 | 237.8045 | 232.9158 | 0 | 1 |
| 2 | pusht_episode_027_step_000 | DINO | DINO | 341.2427 | 346.3792 | 290.1900 | 0 | 0 |
| 2 | pusht_episode_025_step_000 | DINO | DINO | 248.9010 | 248.9010 | 212.1168 | 0 | 0 |
| 2 | pusht_episode_036_step_000 | UMM | DINO | 239.7680 | 326.9386 | 193.2648 | 0 | 0 |
| 2 | pusht_episode_008_step_000 | DINO | UMM | 253.6210 | 208.3793 | 187.7545 | 0 | 0 |
| 2 | pusht_episode_024_step_000 | UMM | UMM | 101.5474 | 71.4483 | 71.4483 | 1 | 0 |
| 3 | pusht_episode_082_step_000 | DINO | DINO | 108.6184 | 108.6184 | 108.6184 | 0 | 0 |
| 3 | pusht_episode_001_step_000 | DINO | DINO | 132.0350 | 165.1152 | 131.9604 | 0 | 0 |
| 3 | pusht_episode_099_step_000 | DINO | DINO | 150.5418 | 150.5418 | 150.5418 | 0 | 0 |
| 3 | pusht_episode_033_step_000 | DINO | DINO | 179.5691 | 179.5691 | 154.4510 | 0 | 0 |
| 3 | pusht_episode_091_step_000 | DINO | DINO | 159.1865 | 193.4948 | 159.1865 | 0 | 1 |
| 3 | pusht_episode_020_step_000 | DINO | UMM | 247.9845 | 116.7836 | 116.7836 | 1 | 0 |
| 3 | pusht_episode_005_step_000 | UMM | DINO | 199.3289 | 217.8610 | 169.2858 | 0 | 0 |
| 3 | pusht_episode_070_step_000 | UMM | UMM | 196.1632 | 189.9287 | 143.9095 | 0 | 0 |
| 3 | pusht_episode_075_step_000 | DINO | UMM | 201.2567 | 191.1233 | 160.0368 | 0 | 0 |
| 3 | pusht_episode_085_step_000 | UMM | DINO | 160.7270 | 219.7737 | 157.0029 | 0 | 0 |
| 3 | pusht_episode_078_step_000 | DINO | DINO | 170.4047 | 195.9047 | 170.4047 | 0 | 1 |
| 3 | pusht_episode_035_step_000 | DINO | DINO | 179.9982 | 187.7890 | 179.9982 | 0 | 1 |
| 3 | pusht_episode_031_step_000 | UMM | DINO | 140.0778 | 195.9417 | 140.0778 | 0 | 1 |
| 3 | pusht_episode_043_step_000 | DINO | UMM | 107.9815 | 106.7329 | 106.7329 | 1 | 0 |
| 4 | pusht_episode_019_step_000 | DINO | UMM | 185.8445 | 168.4468 | 148.9761 | 0 | 0 |
| 4 | pusht_episode_090_step_000 | UMM | UMM | 101.3646 | 61.0930 | 61.0930 | 1 | 0 |
| 4 | pusht_episode_000_step_000 | DINO | DINO | 213.6956 | 257.4475 | 213.6956 | 0 | 1 |
| 4 | pusht_episode_022_step_000 | DINO | UMM | 201.7601 | 165.6405 | 131.4226 | 0 | 0 |
| 4 | pusht_episode_018_step_000 | UMM | DINO | 101.9020 | 145.6140 | 101.9020 | 0 | 1 |
| 4 | pusht_episode_038_step_000 | DINO | UMM | 123.7327 | 83.3282 | 83.3282 | 1 | 0 |
| 4 | pusht_episode_072_step_000 | DINO | DINO | 164.2250 | 164.2250 | 164.2250 | 0 | 0 |
| 4 | pusht_episode_051_step_000 | UMM | DINO | 73.4944 | 73.4944 | 73.4944 | 0 | 0 |
| 4 | pusht_episode_009_step_000 | UMM | UMM | 152.8087 | 106.8235 | 81.0742 | 0 | 0 |
| 4 | pusht_episode_055_step_000 | DINO | DINO | 161.2197 | 161.2197 | 161.2197 | 0 | 0 |
| 4 | pusht_episode_050_step_000 | DINO | DINO | 168.3454 | 168.3454 | 168.3454 | 0 | 0 |
| 4 | pusht_episode_046_step_000 | DINO | UMM | 198.0916 | 145.3145 | 145.3145 | 1 | 0 |
| 4 | pusht_episode_058_step_000 | DINO | UMM | 163.2877 | 161.3049 | 161.3049 | 1 | 0 |
| 4 | pusht_episode_042_step_000 | DINO | DINO | 202.6690 | 202.6690 | 202.3617 | 0 | 0 |
| 5 | pusht_episode_069_step_000 | UMM | DINO | 212.6912 | 226.9488 | 141.3929 | 0 | 0 |
| 5 | pusht_episode_073_step_000 | DINO | DINO | 249.5802 | 249.5802 | 231.9325 | 0 | 0 |
| 5 | pusht_episode_062_step_000 | UMM | UMM | 257.5710 | 219.0417 | 202.2568 | 0 | 0 |
| 5 | pusht_episode_023_step_000 | UMM | UMM | 237.5745 | 214.9687 | 196.1541 | 0 | 0 |
| 5 | pusht_episode_041_step_000 | DINO | DINO | 253.9696 | 280.5926 | 253.9696 | 0 | 1 |
| 5 | pusht_episode_017_step_000 | UMM | DINO | 171.7222 | 235.8356 | 171.7222 | 0 | 1 |
| 5 | pusht_episode_034_step_000 | UMM | DINO | 214.8587 | 300.1569 | 214.8587 | 0 | 1 |
| 5 | pusht_episode_053_step_000 | UMM | UMM | 216.0098 | 193.9919 | 178.1310 | 0 | 0 |
| 5 | pusht_episode_094_step_000 | DINO | DINO | 79.3949 | 79.3949 | 79.3949 | 0 | 0 |
| 5 | pusht_episode_097_step_000 | DINO | DINO | 112.1275 | 112.1275 | 73.2781 | 0 | 0 |
| 5 | pusht_episode_081_step_000 | DINO | DINO | 232.5064 | 232.5064 | 101.1547 | 0 | 0 |
| 5 | pusht_episode_092_step_000 | UMM | DINO | 248.9463 | 304.7403 | 248.9463 | 0 | 1 |
| 5 | pusht_episode_057_step_000 | DINO | UMM | 150.1747 | 138.2555 | 138.2555 | 1 | 0 |
| 5 | pusht_episode_007_step_000 | DINO | DINO | 276.1435 | 276.1435 | 234.5503 | 0 | 0 |
| 6 | pusht_episode_066_step_000 | DINO | DINO | 111.9965 | 196.5079 | 111.9965 | 0 | 1 |
| 6 | pusht_episode_060_step_000 | DINO | UMM | 205.5506 | 201.1260 | 201.1260 | 1 | 0 |
| 6 | pusht_episode_068_step_000 | DINO | DINO | 172.6805 | 178.5998 | 172.6805 | 0 | 1 |
| 6 | pusht_episode_015_step_000 | DINO | DINO | 139.9402 | 161.5770 | 139.9402 | 0 | 1 |
| 6 | pusht_episode_067_step_000 | DINO | DINO | 182.7628 | 227.4799 | 182.7628 | 0 | 1 |
| 6 | pusht_episode_026_step_000 | DINO | DINO | 190.1991 | 190.1991 | 190.1991 | 0 | 0 |
| 6 | pusht_episode_037_step_000 | DINO | DINO | 144.1587 | 144.1587 | 120.6637 | 0 | 0 |
| 6 | pusht_episode_002_step_000 | DINO | DINO | 135.3626 | 135.3626 | 120.5597 | 0 | 0 |
| 6 | pusht_episode_063_step_000 | UMM | DINO | 246.9017 | 255.5326 | 246.9017 | 0 | 1 |
| 6 | pusht_episode_048_step_000 | DINO | UMM | 144.6869 | 140.1977 | 140.1977 | 1 | 0 |
| 6 | pusht_episode_049_step_000 | DINO | DINO | 142.0459 | 142.0459 | 136.8540 | 0 | 0 |
| 6 | pusht_episode_096_step_000 | UMM | UMM | 115.5601 | 84.8829 | 84.8829 | 1 | 0 |
| 6 | pusht_episode_029_step_000 | UMM | DINO | 274.3109 | 470.6542 | 274.3109 | 0 | 1 |
| 6 | pusht_episode_013_step_000 | UMM | DINO | 145.3166 | 145.9289 | 112.2681 | 0 | 0 |

## Interpretation

- This directly tests the UMM-as-world-model claim: UMM should intervene mainly when DINO's static visual choice is dynamically unreliable.
- If this gate improves oracle matches but not mean distance, the proposal should frame UMM as a planning uncertainty and candidate-identification module.
- If this gate is unstable, the next required step is richer generative rollout supervision rather than more static representation fusion.
