# Preservation record: role-swap reconstruction and critical-window archives

These archives are kept outside Git. They were delivered to the owner on 2026-11-25 as session attachments, the same route used for the factorial archives (`runs/concept_factorial/PRESERVATION.md`). The originals are retained in the container, in `/home/user/swap_archives` and `/home/user/swap_window_archives`.

**Owner-side verification: pending.** Checksums and manifests prove integrity; they do not themselves preserve the tensors. Until the owner verifies the delivered copies, the container copies are the only verified copies.

## Verify the delivered copies

1. Keep the two sets in **separate folders**, because the window archives reuse filenames.
2. Verify each folder against its own checksum file:

```
python runs/concept_factorial/verify_archives.py <reconstruction-folder> runs/swap_recon/archives/SHA256SUMS_reconstruction
python runs/concept_factorial/verify_archives.py <window-folder> runs/swap_recon/archives/SHA256SUMS_window
```

3. Optionally, deep-check one archive's contents against its per-run manifest:

```
python runs/concept_factorial/verify_archives.py --contents <archive.tar.xz> runs/swap_recon/<run>/manifest.json
```

Window manifests are in `runs/swap_recon/window/<run>/`.

The checksums agree with every per-run manifest's archive digest (0 mismatches across 41 archives, checked 2026-11-25).

## Reconstruction archives

36 runs, 2,196 checkpoints, 2.83 GB. Every endpoint is bitwise identical to the historical networks (`qualification.json`).

| run | sha256 | delivery id | owner check |
|---|---|---|---|
| acc_only_s300.tar.xz | `31761eda42e98e87be67871ce15677b7adca773fb7cbf0c5bb5ba197e42e008f` | 7a0e4f34-9340-4970-aea3-bba2b7fec145 | pending |
| acc_only_s301.tar.xz | `245d6aa6bd1afe3f8842bdbcf0213b53e0f7dd1dd75b28b4278e0a10df7e91a8` | a452518e-b694-47fe-aae7-8baeb576b0b3 | pending |
| acc_only_s302.tar.xz | `6a5d0f48b9c47be8b99de7a0093974312266bcb05b691333c2c7984e571154ec` | 47c04572-2d29-4cae-a549-1af6a1897105 | pending |
| acc_only_s303.tar.xz | `c8e7e1571ab1a200b0df7ccc6f287f0e17269d04c5a1a2259e915f63f7fad2dc` | 2cedc842-cc9d-4776-9fa8-08db8213d265 | pending |
| acc_only_s304.tar.xz | `33998a1f9f1457e48483d87feea1842a129134dfa06da5369731d4415c942159` | 7e8d03e7-fc4a-436e-967f-b66b65114d89 | pending |
| acc_only_s305.tar.xz | `c044112126d5117505f4878c49779ce17d3e3f5b097633e9f6b1e1a492a5336d` | 990ff4e8-9382-43b3-b148-55b7d146e35f | pending |
| acc_only_s306.tar.xz | `bd1ce52606cf544821265f3930447d898166976b28fcfc7548377ed78071c5c4` | a12f38ec-d32b-4350-aa11-02e2a9c35b76 | pending |
| acc_only_s307.tar.xz | `a88766c89c5c4542bcad6444ca0be20494fbf9d976f82ddbd063f4242bc4d792` | 3655c57c-b8ea-444e-a3c0-b323ec20a2af | pending |
| acc_only_s308.tar.xz | `29d56ce289fda6c383053765c90836f27c1ab6f47bf2561655c54bdfea1dc2da` | 707bccce-3116-45f2-a3aa-a610debc9d68 | pending |
| acc_only_s309.tar.xz | `5a621bb1bef049e1a6c40ca7a47f4a6d95347e21fb9bac7ac8505a982fecd654` | 1e5295ba-9bb9-4d40-ae58-75cf381dd02f | pending |
| acc_only_s310.tar.xz | `f2d85260a481ef344503bae847a7fa7984621064960eca3e3f3b6f3381f480bd` | b43e5a69-2e92-48eb-98e4-9f8a809ed142 | pending |
| acc_only_s311.tar.xz | `c0dae5029eb5e92cffd0d5a23a37229c4c80b76101b8bc67ced0a1012c4bcab1` | 9081313c-219e-4249-9efa-452e7f3073fc | pending |
| mass_only_s300.tar.xz | `e473d2d1ff60736adf67c7eab0f5e725cfa260b43226d548d37732eb1e09df58` | 81ad2602-6fd5-41cb-8e61-05214641d9d1 | pending |
| mass_only_s301.tar.xz | `d5d1396fe8f50809d3e43336e130094310c14d072c7a5ad7ccb25ab48bc3364a` | 953d50b6-8c7c-406b-9085-c24527b975c1 | pending |
| mass_only_s302.tar.xz | `0b7bafe71a4c0958d9e13bac183a9186c066016c87591ffb0fe6aeb93f4b9dd1` | e6b7c293-4699-41d8-98ff-bce0b6a4d9c0 | pending |
| mass_only_s303.tar.xz | `dad310bebfb25bcc4221e5b647edf986570935280d299cf83720d3ea7c3b78d1` | 92a0b91d-b6bf-4d2e-bfe1-94273a02a261 | pending |
| mass_only_s304.tar.xz | `93e9b0653eb41a2ed4a17b5d362a40772afe4ebf5d51f712bc4f3074eac7c4fa` | 8a7af96b-c370-41ae-b285-70b424faed62 | pending |
| mass_only_s305.tar.xz | `a6e6ceb6e85f764af8c1d9715570c4987406174069b3499d724b636aac037b90` | bf10ddf5-0d2e-432f-98b6-960683e442b4 | pending |
| mass_only_s306.tar.xz | `edd25655f273f3d7dfea2f74593b2f2aad6a07712c73382eec0abf0fce030543` | e9c580c4-0870-4a77-a9ed-b51c5e2aed79 | pending |
| mass_only_s307.tar.xz | `caa3dad9fade2b4fe7ef40ce5d45f40b7585b9c35fe583d8cac26218c3a3157d` | 70e72e3a-3540-46db-8821-6d38f0221389 | pending |
| mass_only_s308.tar.xz | `3f1b2a76e3a028402634ede32b7610c20f997f7533661da00947af0a85fe13fa` | cd54e43d-1820-48a6-8c3e-039996b412f7 | pending |
| mass_only_s309.tar.xz | `70868ba24934f92adfced213676e61261df6fdae40120d84a7cfb08faf1ca86d` | 40485bc4-920b-4c53-a4e5-b18ddae467e9 | pending |
| mass_only_s310.tar.xz | `1913ebdd9d9380c877d253eaf08624e808a3809e8cfc3b91092a49d2f46dd27f` | 26dad84b-026f-4acc-a900-d5fa405a6029 | pending |
| mass_only_s311.tar.xz | `3323c4d409d15dfdebe13bbf4eba14c26a56180fd9a4ea517479839fa45d9003` | 292d915e-f83e-4ff4-b3ae-26e5dda064fd | pending |
| neither_s300.tar.xz | `d33b2553ba82026b9002ba7dc633e3025de4519d551eee158b580c098c4e15d8` | 4f9f040f-9d7e-4237-8009-e0a33950147a | pending |
| neither_s301.tar.xz | `3b3196e7b014865ee165c9d94b618c9f09f68d15c9e5d2313e0189d3087bdc68` | 239ea251-0130-42ad-9d72-a0671554d4b4 | pending |
| neither_s302.tar.xz | `f76658084bfd9f3eda7becbc42d92aa4c62ac4e99452f5bab7b0aef8a73e390c` | ec007472-5ac7-4c28-bfb0-6d1535174da1 | pending |
| neither_s303.tar.xz | `cf487bd5238a606d0a46d84f4a09b41ec2e7065ad6a36866c2f686791d2e1afa` | d53ed707-a136-488b-89ee-381f9b46f9b4 | pending |
| neither_s304.tar.xz | `51f78efbfebf68458a901c0bf8da894edb2199a36e5d8863b8f80373f6935cc2` | 2ee9dba3-3af1-4205-a3b4-628ffafe456b | pending |
| neither_s305.tar.xz | `69c2ccd3212b4091fdda377df4abe6d0d18300c1cc085e191f490ea7e35b5569` | 533904f5-a09e-4857-af2a-25bcb7406e05 | pending |
| neither_s306.tar.xz | `951c91512feed91b8bf7990de75115f0260a01e0a5688fb6ecaff21e4ebcc8c2` | 86a395c4-3a03-4e98-8dba-e1fedb005b2d | pending |
| neither_s307.tar.xz | `b407e07c2a1c5f0cb0db3fb01d1756d38501ec4aaf4a45db4a2b6fa8a0beba82` | 9b35f73d-5e4a-4518-af99-9d6bde19f253 | pending |
| neither_s308.tar.xz | `81360fcad9f5708c443225ae3636343caaca09ea332fadfb786f2774b9074553` | 834c1d32-e15b-4eec-9649-47763541c226 | pending |
| neither_s309.tar.xz | `c2f6f468ae5c57d1239232962f1fdd59ff6023a73e92771a5d9bb7bae2bd2f6a` | 6d0291d7-71a2-4bd7-b896-6f34be7656d6 | pending |
| neither_s310.tar.xz | `fd5148d7c9cade7cff84320a070086fe62e2ff14f9793cbbe92a3480a407ba7a` | 584f186f-5ace-4892-ba81-98f3b942451c | pending |
| neither_s311.tar.xz | `89c371ef30514e04e5aa0312ea3ca25531894bd6fab02f0336c02c2f36f03160` | 75241d82-fe29-4423-a6ec-4d5c273d9af7 | pending |

## Critical-window archives

5 runs, 705 states at 10-step resolution, steps 600–2,000. Each run is bitwise-qualified at steps 1,000 and 2,000.

| run | sha256 | delivery id | owner check |
|---|---|---|---|
| mass_only_s304.tar.xz | `3bb7af980c8b5add67674606e5177dbe5b2dcafe13641c2c6ccf1d0464787468` | 48811355-baad-4e0d-925e-5e2336bd9aff | pending |
| mass_only_s305.tar.xz | `6eb1c11bb0916c708d9bb6f1ffe1fd1274897f5471527c8b48a10ad8917333ad` | d369c83c-a874-4623-8ee9-64e778f40ec7 | pending |
| mass_only_s307.tar.xz | `876972d47d172f2676c90b5278a77f3efe310bacd6f4fd4b654aea8774137c8e` | 0bf041ff-0962-426b-aa5a-0b407f36307a | pending |
| mass_only_s310.tar.xz | `934546044cb1590feb27de355151ba740bd6aba8f57c8f40e5ae277ed9031174` | 4236ff2b-8dac-4bf2-ab44-4810b28093af | pending |
| mass_only_s311.tar.xz | `5ce1ac2cb2a5e85561aea07d3482af84beddd0f15b78dc45eed51f35687a73c5` | 455bcb55-19d4-4b53-881b-bedc2af56db8 | pending |

## Checksum files

- `SHA256SUMS_reconstruction`: delivery id 4bf2f8d9-b604-43f1-ba46-21b14bc526a9
- `SHA256SUMS_window`: delivery id 869e45fa-f8e4-4800-8665-cf5f75071e9f
