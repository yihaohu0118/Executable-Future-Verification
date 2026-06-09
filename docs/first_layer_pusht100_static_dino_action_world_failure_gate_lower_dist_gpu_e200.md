# DINO-Failure-Aware UMM World-Model Gate

## Bottom Line

This trains a fold-wise binary gate that decides whether a case should stay with the static DINO selector or be overridden by a UMM world-model selector. The gate uses only non-oracle confidence, planner, and action-conditioned rollout features at test time.

## Result

| Objective | Mean dist | Better | Oracle match | Pair acc | Expert acc | Gate acc | Overrides |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lower_dist | 182.0821 | 45/100 | 32/100 | 0.6067 | 0.6791 | 0.5900 | 37/100 |

## Case Decisions

| Fold | Case | Pred | Target | DINO dist | UMM dist | Oracle dist | Rescue | Loss |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | pusht_episode_084_step_000 | DINO | DINO | 133.5991 | 133.5991 | 128.9333 | 0 | 0 |
| 0 | pusht_episode_086_step_000 | UMM | DINO | 70.1992 | 99.1527 | 70.1992 | 0 | 1 |
| 0 | pusht_episode_065_step_000 | DINO | DINO | 245.3156 | 245.3156 | 166.1039 | 0 | 0 |
| 0 | pusht_episode_045_step_000 | DINO | DINO | 193.2041 | 193.2041 | 150.8250 | 0 | 0 |
| 0 | pusht_episode_061_step_000 | DINO | UMM | 100.5978 | 70.9816 | 70.9816 | 1 | 0 |
| 0 | pusht_episode_010_step_000 | DINO | DINO | 244.2671 | 323.2168 | 244.2671 | 0 | 1 |
| 0 | pusht_episode_093_step_000 | UMM | UMM | 164.1270 | 148.9881 | 121.7164 | 0 | 0 |
| 0 | pusht_episode_087_step_000 | DINO | DINO | 240.2391 | 240.2391 | 240.2391 | 0 | 0 |
| 0 | pusht_episode_059_step_000 | UMM | DINO | 252.5808 | 298.0004 | 252.5808 | 0 | 1 |
| 0 | pusht_episode_098_step_000 | DINO | DINO | 244.0301 | 244.0301 | 208.4497 | 0 | 0 |
| 0 | pusht_episode_088_step_000 | UMM | DINO | 115.2581 | 194.6293 | 115.2581 | 0 | 1 |
| 0 | pusht_episode_076_step_000 | UMM | UMM | 125.9020 | 86.1742 | 86.1742 | 1 | 0 |
| 0 | pusht_episode_003_step_000 | UMM | DINO | 133.1285 | 133.8201 | 133.1285 | 0 | 1 |
| 0 | pusht_episode_095_step_000 | UMM | DINO | 119.6308 | 166.6705 | 119.6308 | 0 | 1 |
| 0 | pusht_episode_089_step_000 | DINO | DINO | 119.1202 | 119.1202 | 119.1202 | 0 | 0 |
| 1 | pusht_episode_006_step_000 | UMM | DINO | 91.2389 | 116.6393 | 91.2389 | 0 | 1 |
| 1 | pusht_episode_030_step_000 | UMM | DINO | 137.4281 | 162.2721 | 104.5054 | 0 | 0 |
| 1 | pusht_episode_047_step_000 | DINO | DINO | 116.5279 | 116.5279 | 116.5279 | 0 | 0 |
| 1 | pusht_episode_079_step_000 | UMM | DINO | 168.7012 | 181.8283 | 168.7012 | 0 | 1 |
| 1 | pusht_episode_071_step_000 | UMM | DINO | 184.2835 | 247.9806 | 155.8825 | 0 | 0 |
| 1 | pusht_episode_011_step_000 | DINO | DINO | 328.7305 | 328.7305 | 291.4912 | 0 | 0 |
| 1 | pusht_episode_021_step_000 | DINO | UMM | 224.1717 | 218.4669 | 218.4669 | 1 | 0 |
| 1 | pusht_episode_083_step_000 | DINO | DINO | 102.4738 | 124.9349 | 102.4738 | 0 | 1 |
| 1 | pusht_episode_056_step_000 | UMM | UMM | 148.9958 | 135.4677 | 135.4677 | 1 | 0 |
| 1 | pusht_episode_016_step_000 | UMM | UMM | 208.0284 | 168.9526 | 168.9526 | 1 | 0 |
| 1 | pusht_episode_004_step_000 | DINO | DINO | 123.6531 | 123.6531 | 114.7030 | 0 | 0 |
| 1 | pusht_episode_080_step_000 | UMM | UMM | 173.7162 | 164.7224 | 162.0198 | 0 | 0 |
| 1 | pusht_episode_028_step_000 | UMM | UMM | 203.3848 | 194.9867 | 194.9867 | 1 | 0 |
| 1 | pusht_episode_039_step_000 | DINO | DINO | 155.4751 | 155.4751 | 144.0669 | 0 | 0 |
| 1 | pusht_episode_074_step_000 | DINO | UMM | 278.9452 | 255.2733 | 251.5667 | 0 | 0 |
| 2 | pusht_episode_012_step_000 | UMM | DINO | 138.7300 | 152.1405 | 122.8260 | 0 | 0 |
| 2 | pusht_episode_054_step_000 | UMM | DINO | 185.0785 | 200.3871 | 185.0785 | 0 | 1 |
| 2 | pusht_episode_077_step_000 | UMM | UMM | 194.2460 | 173.0203 | 173.0203 | 1 | 0 |
| 2 | pusht_episode_032_step_000 | UMM | UMM | 252.7009 | 190.2089 | 190.2089 | 1 | 0 |
| 2 | pusht_episode_052_step_000 | DINO | DINO | 152.5085 | 152.5085 | 152.5085 | 0 | 0 |
| 2 | pusht_episode_040_step_000 | DINO | UMM | 184.8285 | 175.5304 | 153.3040 | 0 | 0 |
| 2 | pusht_episode_064_step_000 | DINO | DINO | 109.7803 | 109.7803 | 109.7803 | 0 | 0 |
| 2 | pusht_episode_014_step_000 | DINO | DINO | 173.0271 | 203.7301 | 173.0271 | 0 | 1 |
| 2 | pusht_episode_044_step_000 | UMM | DINO | 232.9158 | 278.8478 | 232.9158 | 0 | 1 |
| 2 | pusht_episode_027_step_000 | DINO | DINO | 341.2427 | 341.2427 | 290.1900 | 0 | 0 |
| 2 | pusht_episode_025_step_000 | DINO | DINO | 259.1443 | 259.1443 | 212.1168 | 0 | 0 |
| 2 | pusht_episode_036_step_000 | UMM | UMM | 237.3006 | 193.2648 | 193.2648 | 1 | 0 |
| 2 | pusht_episode_008_step_000 | DINO | UMM | 225.1886 | 208.3793 | 187.7545 | 0 | 0 |
| 2 | pusht_episode_024_step_000 | DINO | UMM | 101.5474 | 71.4483 | 71.4483 | 1 | 0 |
| 3 | pusht_episode_082_step_000 | DINO | DINO | 108.6184 | 108.6184 | 108.6184 | 0 | 0 |
| 3 | pusht_episode_001_step_000 | UMM | DINO | 132.0350 | 165.1152 | 131.9604 | 0 | 0 |
| 3 | pusht_episode_099_step_000 | UMM | UMM | 180.8266 | 150.5418 | 150.5418 | 1 | 0 |
| 3 | pusht_episode_033_step_000 | DINO | DINO | 179.5691 | 179.5691 | 154.4510 | 0 | 0 |
| 3 | pusht_episode_091_step_000 | UMM | UMM | 176.7911 | 159.1865 | 159.1865 | 1 | 0 |
| 3 | pusht_episode_020_step_000 | UMM | DINO | 146.0936 | 247.9845 | 116.7836 | 0 | 0 |
| 3 | pusht_episode_005_step_000 | DINO | DINO | 179.9088 | 199.3289 | 169.2858 | 0 | 0 |
| 3 | pusht_episode_070_step_000 | DINO | DINO | 196.1632 | 196.1632 | 143.9095 | 0 | 0 |
| 3 | pusht_episode_075_step_000 | UMM | UMM | 201.2567 | 191.1233 | 160.0368 | 0 | 0 |
| 3 | pusht_episode_085_step_000 | DINO | DINO | 160.7270 | 160.7270 | 157.0029 | 0 | 0 |
| 3 | pusht_episode_078_step_000 | DINO | DINO | 170.4047 | 170.4047 | 170.4047 | 0 | 0 |
| 3 | pusht_episode_035_step_000 | UMM | DINO | 179.9982 | 198.8993 | 179.9982 | 0 | 1 |
| 3 | pusht_episode_031_step_000 | DINO | DINO | 140.0778 | 140.0778 | 140.0778 | 0 | 0 |
| 3 | pusht_episode_043_step_000 | UMM | UMM | 145.2908 | 106.7329 | 106.7329 | 1 | 0 |
| 4 | pusht_episode_019_step_000 | DINO | UMM | 185.8445 | 168.4468 | 148.9761 | 0 | 0 |
| 4 | pusht_episode_090_step_000 | UMM | UMM | 101.3646 | 61.0930 | 61.0930 | 1 | 0 |
| 4 | pusht_episode_000_step_000 | UMM | DINO | 233.5464 | 254.1759 | 213.6956 | 0 | 0 |
| 4 | pusht_episode_022_step_000 | DINO | DINO | 134.5655 | 134.5655 | 131.4226 | 0 | 0 |
| 4 | pusht_episode_018_step_000 | DINO | DINO | 134.7458 | 145.6140 | 101.9020 | 0 | 0 |
| 4 | pusht_episode_038_step_000 | DINO | UMM | 89.4965 | 83.3282 | 83.3282 | 1 | 0 |
| 4 | pusht_episode_072_step_000 | UMM | DINO | 164.2250 | 244.2455 | 164.2250 | 0 | 1 |
| 4 | pusht_episode_051_step_000 | DINO | DINO | 73.4944 | 73.4944 | 73.4944 | 0 | 0 |
| 4 | pusht_episode_009_step_000 | UMM | UMM | 152.8087 | 106.8235 | 81.0742 | 0 | 0 |
| 4 | pusht_episode_055_step_000 | DINO | UMM | 192.6203 | 161.2197 | 161.2197 | 1 | 0 |
| 4 | pusht_episode_050_step_000 | DINO | UMM | 198.0497 | 180.9928 | 168.3454 | 0 | 0 |
| 4 | pusht_episode_046_step_000 | DINO | UMM | 198.0916 | 145.3145 | 145.3145 | 1 | 0 |
| 4 | pusht_episode_058_step_000 | DINO | DINO | 161.3049 | 163.2877 | 161.3049 | 0 | 1 |
| 4 | pusht_episode_042_step_000 | DINO | DINO | 202.6690 | 227.8459 | 202.3617 | 0 | 0 |
| 5 | pusht_episode_069_step_000 | UMM | UMM | 212.6912 | 141.3929 | 141.3929 | 1 | 0 |
| 5 | pusht_episode_073_step_000 | DINO | DINO | 249.5802 | 249.5802 | 231.9325 | 0 | 0 |
| 5 | pusht_episode_062_step_000 | DINO | DINO | 202.2568 | 202.2568 | 202.2568 | 0 | 0 |
| 5 | pusht_episode_023_step_000 | DINO | UMM | 214.9687 | 196.1541 | 196.1541 | 1 | 0 |
| 5 | pusht_episode_041_step_000 | DINO | UMM | 331.9012 | 253.9696 | 253.9696 | 1 | 0 |
| 5 | pusht_episode_017_step_000 | DINO | DINO | 171.7222 | 171.7222 | 171.7222 | 0 | 0 |
| 5 | pusht_episode_034_step_000 | UMM | DINO | 268.7889 | 300.1569 | 214.8587 | 0 | 0 |
| 5 | pusht_episode_053_step_000 | UMM | UMM | 216.0098 | 193.9919 | 178.1310 | 0 | 0 |
| 5 | pusht_episode_094_step_000 | DINO | DINO | 79.3949 | 79.3949 | 79.3949 | 0 | 0 |
| 5 | pusht_episode_097_step_000 | UMM | DINO | 73.2781 | 112.1275 | 73.2781 | 0 | 1 |
| 5 | pusht_episode_081_step_000 | DINO | UMM | 137.0757 | 101.1547 | 101.1547 | 1 | 0 |
| 5 | pusht_episode_092_step_000 | UMM | DINO | 248.9463 | 299.9096 | 248.9463 | 0 | 1 |
| 5 | pusht_episode_057_step_000 | DINO | DINO | 150.1747 | 150.1747 | 138.2555 | 0 | 0 |
| 5 | pusht_episode_007_step_000 | DINO | DINO | 334.3693 | 334.3693 | 234.5503 | 0 | 0 |
| 6 | pusht_episode_066_step_000 | DINO | UMM | 196.5079 | 111.9965 | 111.9965 | 1 | 0 |
| 6 | pusht_episode_060_step_000 | DINO | DINO | 210.7304 | 210.7304 | 201.1260 | 0 | 0 |
| 6 | pusht_episode_068_step_000 | DINO | UMM | 249.8103 | 178.5998 | 172.6805 | 0 | 0 |
| 6 | pusht_episode_015_step_000 | DINO | DINO | 139.9402 | 238.0537 | 139.9402 | 0 | 1 |
| 6 | pusht_episode_067_step_000 | DINO | UMM | 248.4499 | 227.4799 | 182.7628 | 0 | 0 |
| 6 | pusht_episode_026_step_000 | DINO | UMM | 258.2984 | 190.1991 | 190.1991 | 1 | 0 |
| 6 | pusht_episode_037_step_000 | DINO | DINO | 120.6637 | 158.7815 | 120.6637 | 0 | 1 |
| 6 | pusht_episode_002_step_000 | DINO | UMM | 135.3626 | 122.4806 | 120.5597 | 0 | 0 |
| 6 | pusht_episode_063_step_000 | DINO | UMM | 263.3677 | 246.9017 | 246.9017 | 1 | 0 |
| 6 | pusht_episode_048_step_000 | DINO | DINO | 140.1977 | 140.1977 | 140.1977 | 0 | 0 |
| 6 | pusht_episode_049_step_000 | DINO | DINO | 142.0459 | 142.0459 | 136.8540 | 0 | 0 |
| 6 | pusht_episode_096_step_000 | DINO | UMM | 86.1383 | 84.8829 | 84.8829 | 1 | 0 |
| 6 | pusht_episode_029_step_000 | DINO | DINO | 274.3109 | 274.3109 | 274.3109 | 0 | 0 |
| 6 | pusht_episode_013_step_000 | DINO | DINO | 145.3166 | 145.9289 | 112.2681 | 0 | 0 |

## Interpretation

- This directly tests the UMM-as-world-model claim: UMM should intervene mainly when DINO's static visual choice is dynamically unreliable.
- If this gate improves oracle matches but not mean distance, the proposal should frame UMM as a planning uncertainty and candidate-identification module.
- If this gate is unstable, the next required step is richer generative rollout supervision rather than more static representation fusion.
