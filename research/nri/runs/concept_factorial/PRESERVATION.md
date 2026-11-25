# Preservation record — concept factorial checkpoint archives

Updated 2026-11-25. **Status: delivered to the owner, owner verification pending. Originals retained.**

| Copy | Location | Verification |
|---|---|---|
| Original | `/home/user/nri_archives/` in the Claude Code cloud session container (ephemeral) | all 96 re-verified against `SHA256SUMS` on 2026-11-25: 96 OK (`sha256sum -c`); deep content check of `wd1.0_C_s2004` against its manifest: 673/673 files match |
| Owner copy | delivered as file attachments in the Claude Code session (one attachment per archive; delivery ids below) | **pending**: owner runs `python verify_archives.py <folder>` → must report 96 OK |
| GitHub release | not created | this session has no tool that can create releases or upload assets; the owner can attach the verified owner copies to a release (≈ 160 MB each, below GitHub's 2 GB per-asset limit) |

Total: 96 archives, 15,521,974,056 bytes. 5,856 checkpoints (61 per run). What Git alone preserves: manifests,
per-file SHA-256, audit summaries, interval logs, code — **not the tensors**. Bit-exact regeneration from Git was
verified for one network over 300 steps in this environment (see `ARCHIVES.md`); that is a reproduction route,
not preservation of the original files.

## Restore and verify

1. Put all `.tar.xz` files, `SHA256SUMS` and `verify_archives.py` in one folder.
2. `python verify_archives.py .` → 96 OK. Any MISMATCH/MISSING: re-download that file.
3. Optional deep check: `python verify_archives.py --contents wd1.0_C_s2004.tar.xz wd1.0_C_s2004/manifest.json`
   (manifest from this Git folder).
4. Extract: `tar -xJf <run>.tar.xz` → `<run>/ckpt/step_00000 … step_06000/` (model, aux, optimizer, RNG,
   update decomposition, audit tensors) and `<run>/interval.jsonl`.

## Inventory

| run | archive | bytes | sha256 | delivery id | owner verification |
|---|---|---|---|---|---|
| wd0.1_A_s2001 | `wd0.1_A_s2001.tar.xz` | 159,920,452 | `4daae94496848b1d728f54689f0e78b2112493f4eb4e7fbeebb5eb462e2e7e07` | `ea6547cd-e693-430d-a804-d8385d742807` | pending owner check |
| wd0.1_A_s2002 | `wd0.1_A_s2002.tar.xz` | 159,868,196 | `81579129cf178e8fb3c157d4ed3f20fd6a0734c8daa48632ff3f849766ea04bc` | `c527d4cf-8e54-4751-8cd9-806a1570c465` | pending owner check |
| wd0.1_A_s2003 | `wd0.1_A_s2003.tar.xz` | 160,111,484 | `2691627b6da418169e7a0e860dd669289b8a80a3bf3e90a0158c7c53f53eb9d6` | `026328bb-a424-4afe-a40f-abd190b718e6` | pending owner check |
| wd0.1_A_s2004 | `wd0.1_A_s2004.tar.xz` | 160,089,348 | `2535136e6c1f201b61fbd1b3e246c0b9d2add56bb468d89966ec424ab4112ac2` | `7bcd1d8c-ed10-47f7-8994-56143046bd29` | pending owner check |
| wd0.1_A_s2005 | `wd0.1_A_s2005.tar.xz` | 160,102,904 | `200f0d1819fac8aa94094e1bec66f7b5573f9a55f0be11d48f0480cd9b7bc177` | `b3de79cc-6d6b-451c-b2f1-887611edc9ce` | pending owner check |
| wd0.1_A_s2006 | `wd0.1_A_s2006.tar.xz` | 160,156,860 | `80b08a759231bb746a8e63a3f0a9106ae3c77973a84b7fbc328efc1f93047f73` | `58143d12-6542-4d36-8db8-a562e967e696` | pending owner check |
| wd0.1_A_s2007 | `wd0.1_A_s2007.tar.xz` | 160,079,780 | `4023cf812b107583179be277cbe12a1252584dc0367b74fe86f051a08e0dc035` | `39a3c709-6d67-44d6-a27d-313d6d955946` | pending owner check |
| wd0.1_A_s2008 | `wd0.1_A_s2008.tar.xz` | 159,937,060 | `7008c9adb20ae0d9b3245ecdd55bdb37284df0d36cf4d70bcb76f854ff11e41d` | `505c7c67-821b-424c-b147-52b1140d69b7` | pending owner check |
| wd0.1_A_s2009 | `wd0.1_A_s2009.tar.xz` | 160,127,588 | `6b49f641e9ad491b8cce4fe1b71e6c504963df25b804e60abbf25fb521972e7b` | `ee419707-e541-41ee-8166-807722331c3f` | pending owner check |
| wd0.1_A_s2010 | `wd0.1_A_s2010.tar.xz` | 160,138,172 | `3faec0da819add67dedd7ac10e82ad3afb47843184edf10971b00a39bd5b9c10` | `e6d5a9a3-38d6-4323-af83-6330ff626939` | pending owner check |
| wd0.1_A_s2011 | `wd0.1_A_s2011.tar.xz` | 160,080,820 | `f8e7e94ae2f30b2e74236ab431609262eefa053ef2e7234aa65df781e54066f2` | `1bfeb0b6-fe01-4cc2-9c80-cec733784797` | pending owner check |
| wd0.1_A_s2012 | `wd0.1_A_s2012.tar.xz` | 159,858,788 | `5397eb3526394a09b9bc29f509db6318a97277b46c4c2e75ad178d11ea75b7db` | `4264ad88-383b-4d26-be8e-25083854bacf` | pending owner check |
| wd0.1_B_s2001 | `wd0.1_B_s2001.tar.xz` | 161,245,444 | `556f723d69d6b8bd33ae67046e5e02601deda4d26e49c872ecfef486ed32e768` | `9b108f37-9382-45e7-aad6-0aacd3927101` | pending owner check |
| wd0.1_B_s2002 | `wd0.1_B_s2002.tar.xz` | 161,138,904 | `0b8455507e73f118f7d43e0148bacab8d25933fe824c4191e54214fc46cdb674` | `4927123c-17cb-49b4-8871-8665d18c2498` | pending owner check |
| wd0.1_B_s2003 | `wd0.1_B_s2003.tar.xz` | 161,097,484 | `3b99adb2fa4eea163c0c9bf725923f90c7d4f2bd608c811bfcf40c570a87358a` | `2ce18031-b281-452b-9eb2-49ea7d90a449` | pending owner check |
| wd0.1_B_s2004 | `wd0.1_B_s2004.tar.xz` | 161,124,016 | `9bcd273f8eb1e4f3de41e41ed314bf687f1078958b63178e7f58b8283b143e23` | `6666d52d-cc27-4281-a853-7abd8f35fc74` | pending owner check |
| wd0.1_B_s2005 | `wd0.1_B_s2005.tar.xz` | 161,157,292 | `f5540f4e85b4055027aa0d2b1c30446057ccd730d6c9c9292edd152ba1ce86fb` | `afa8bbc9-b51e-43e9-948c-da752c69aa97` | pending owner check |
| wd0.1_B_s2006 | `wd0.1_B_s2006.tar.xz` | 161,220,516 | `93379d5a2a3aa6ff5b917b970feb6ed0a687223fd5e55ebf667957e55982f131` | `f399c13c-85dc-4e86-b17f-92cfd7bd3828` | pending owner check |
| wd0.1_B_s2007 | `wd0.1_B_s2007.tar.xz` | 161,042,724 | `975f383b051fdf90ad00dc704eb57c990bb6bb0bbfcc3c8fd7da12bd5574837b` | `fe30a4eb-7abf-44ce-93af-e96440982505` | pending owner check |
| wd0.1_B_s2008 | `wd0.1_B_s2008.tar.xz` | 161,174,548 | `08494b2834ad49171b22035e31b65b8cc168264894e5ae1b8c184e5ab9772d3f` | `79325314-630c-4cf3-89b8-97770c1f05a0` | pending owner check |
| wd0.1_B_s2009 | `wd0.1_B_s2009.tar.xz` | 161,167,900 | `84249aa5e1976826b9339ce4c0eecbe8ddbd3989041d70c7f2d73ddebb5bd53a` | `fa5454c2-eb4c-4649-a96f-3824889ea396` | pending owner check |
| wd0.1_B_s2010 | `wd0.1_B_s2010.tar.xz` | 161,211,208 | `8e7785a7dc8c3dc64815295af3f37367801b14298383539ecac84bdfb83f410a` | `ab4eefc1-223e-4733-9bee-133aab1ea50c` | pending owner check |
| wd0.1_B_s2011 | `wd0.1_B_s2011.tar.xz` | 161,310,288 | `80f0e9652f0b90bafe73c82ce8879f10e0b9bbe046ea9c7422d3dea06f6d9d8e` | `8df6d28a-ae0a-4661-853f-ba3482986c78` | pending owner check |
| wd0.1_B_s2012 | `wd0.1_B_s2012.tar.xz` | 161,165,052 | `978fd228fff3edcc0fe6fd2c92e62e9911d9cc6fafa26bbcf836db4aae4245df` | `37117538-4c82-494f-96a1-a2f00a975d50` | pending owner check |
| wd0.1_C_s2001 | `wd0.1_C_s2001.tar.xz` | 162,374,348 | `91e0edfd8a497ebfc4b34e6c552c74636b073be20ed706b6b2c0ac2f71042afa` | `a9dd84ec-f551-49d8-bf5e-d3a2481601bf` | pending owner check |
| wd0.1_C_s2002 | `wd0.1_C_s2002.tar.xz` | 162,516,484 | `82a70fef29477874e3fa8f2ac19df1649fe4ed5cb8a8d8b2d47647656dced315` | `a15d7d60-91da-49ce-a2fa-d38e38a8d93a` | pending owner check |
| wd0.1_C_s2003 | `wd0.1_C_s2003.tar.xz` | 162,368,940 | `fbd2cfe124b3f53f1a9a6295832d5154ae88fecb24d029f3e7aa112870a9f77c` | `a188ab0f-9a62-409a-99d2-5b01d98ccf55` | pending owner check |
| wd0.1_C_s2004 | `wd0.1_C_s2004.tar.xz` | 162,497,716 | `9fa436e975eac496a4457ac1b380b4210c2a7ad2a229439e611318ccd57d0d84` | `3d8d7b9c-8713-498d-9dd7-6280d0630a37` | pending owner check |
| wd0.1_C_s2005 | `wd0.1_C_s2005.tar.xz` | 162,458,296 | `44acd534fe0c321e58f83f7bbf2b160c0236ebb379d8f117aa36a82ce3eed4c7` | `c5fee4b2-f711-4455-8559-b7c58edc95e5` | pending owner check |
| wd0.1_C_s2006 | `wd0.1_C_s2006.tar.xz` | 162,527,088 | `a96c1308d64c9d6569d1ccce2968d460da0c5bdf7794410bd9e0ed6cabadd4fd` | `51db3632-0927-4f51-b31b-96b89e0fa6f7` | pending owner check |
| wd0.1_C_s2007 | `wd0.1_C_s2007.tar.xz` | 162,451,004 | `5cb3e23f3d3fd5e1b621044a6dba4760fb156fd3ef2f07dbfdcb25be71d1969f` | `fc410715-e6f6-4459-a570-ff233134b9aa` | pending owner check |
| wd0.1_C_s2008 | `wd0.1_C_s2008.tar.xz` | 162,476,040 | `696f2c3af9f7f8a856f467f0834e8b7f402b1a83074ea13754ac50ddc09dc602` | `c84ec403-e69b-435e-83da-bbc564668190` | pending owner check |
| wd0.1_C_s2009 | `wd0.1_C_s2009.tar.xz` | 162,482,420 | `0f717490abc224d6f20834cf40ec0c71e2207a280b3dc328efe0c9fbddff9be0` | `004d2f65-d073-4b88-aa63-e30e60614949` | pending owner check |
| wd0.1_C_s2010 | `wd0.1_C_s2010.tar.xz` | 162,467,236 | `3b115208a09f34a001231f3ddcb89be7099e2e9686551a0c99481606dfa7f22f` | `bab44b32-d9fc-4d46-8647-160a90e91be6` | pending owner check |
| wd0.1_C_s2011 | `wd0.1_C_s2011.tar.xz` | 162,648,940 | `d0d2455b0a467d11f283106059b38b7ed7714b8b9df1a39217f80057844de38e` | `be1b7318-f57f-4d21-a05c-74bd32575675` | pending owner check |
| wd0.1_C_s2012 | `wd0.1_C_s2012.tar.xz` | 162,661,120 | `ce3f67f4b50f110955d2fb574fc8962410f42c380522d401337139382a5806b2` | `99092866-9431-4cf7-9d63-c5f4416664b0` | pending owner check |
| wd0.1_D_s2001 | `wd0.1_D_s2001.tar.xz` | 162,394,888 | `304e98d0b4e6d1800dc7d27009af01c1d5afc2184056ef3428a3671f50ff6927` | `ab55e334-201d-4b72-bc6e-e84d3079b384` | pending owner check |
| wd0.1_D_s2002 | `wd0.1_D_s2002.tar.xz` | 162,469,132 | `185b20527d94212e5456aa73e0205d29d932aff8dd7b72461af2697b1fbb716c` | `3447562a-6820-4147-b6b8-168ffbdf6dfb` | pending owner check |
| wd0.1_D_s2003 | `wd0.1_D_s2003.tar.xz` | 162,383,304 | `3fa83eff5de64fbd6e54a6af3f645e004d76ce3378ea4d96e372c0f8e237cf30` | `1301f393-67c1-4529-a141-932503bcba31` | pending owner check |
| wd0.1_D_s2004 | `wd0.1_D_s2004.tar.xz` | 162,381,324 | `61cf1087918df937f1e9523c56b902c5feeb4a9404547d4485c914b3681bc792` | `4a3b7910-7ebc-4f43-abfe-03658c65aff0` | pending owner check |
| wd0.1_D_s2005 | `wd0.1_D_s2005.tar.xz` | 162,371,744 | `d85052a22fef08880a3caa07f9964ffc6a1d9ee6bdd1d9fc1fcfb3eab4eb1489` | `13f711f7-e43e-4159-bf8b-0cdb335a3488` | pending owner check |
| wd0.1_D_s2006 | `wd0.1_D_s2006.tar.xz` | 162,374,312 | `f9fe72cdfef484c45d4b3c483c0b28938527d4f54036a9850af387c35fa74f1e` | `d9e5e918-4cfb-4b93-a75d-35a083cecc7e` | pending owner check |
| wd0.1_D_s2007 | `wd0.1_D_s2007.tar.xz` | 162,396,124 | `5d8fecb2bd79c2eb36ed2e4e15af9f27a0c43e401be16f8f4206ffeb7603cc1b` | `5ba71402-10e4-4d5c-a106-7716d402cd3c` | pending owner check |
| wd0.1_D_s2008 | `wd0.1_D_s2008.tar.xz` | 162,447,596 | `9af41b4f8d99e6cdf79f9de008b48b7264b06ee37129b46f62c1fabf9c63c1ee` | `ecf7b319-ee51-4557-b5f7-ede5cfb931d6` | pending owner check |
| wd0.1_D_s2009 | `wd0.1_D_s2009.tar.xz` | 162,492,744 | `8c433009480033a6538f9143a4232eb8844de72f0e9e2b4473dfb4afe7ac2a6a` | `65b4a97f-8e96-4b6b-baab-c7f7f38c91b2` | pending owner check |
| wd0.1_D_s2010 | `wd0.1_D_s2010.tar.xz` | 162,435,348 | `90a878c29ff7e07e783746e170c0a4553e5225d2485d0ea34be8210bbc40d08c` | `98a80e49-8428-49c0-b71a-093972fdc7f9` | pending owner check |
| wd0.1_D_s2011 | `wd0.1_D_s2011.tar.xz` | 162,554,524 | `03e4ca1297dde504721a7688d28cb4ce2bd046e2c83320727ba85cbe9751f7dd` | `b791e68a-9e94-4dbd-8856-1764b88fc3d9` | pending owner check |
| wd0.1_D_s2012 | `wd0.1_D_s2012.tar.xz` | 162,430,748 | `2833259ed6619f086b52fd0aeafc876ccd3b88f1022fb223be1965c7af839012` | `8aabe70a-7bcb-479d-8322-2b4827fc9b19` | pending owner check |
| wd1.0_A_s2001 | `wd1.0_A_s2001.tar.xz` | 160,686,312 | `580e51bb7f2af26e54af576183f711824605eb92747b0e729921cb184314440e` | `0498b1cc-4d77-4ae9-8b94-f4a2f104bbf5` | pending owner check |
| wd1.0_A_s2002 | `wd1.0_A_s2002.tar.xz` | 160,736,416 | `71dc9730c1d2edc9c1b23f0e886c6f84df9c1cf8eed9e49401ff25a978a5d16f` | `c731e6a3-f089-432f-bd6b-cc04f7583bb3` | pending owner check |
| wd1.0_A_s2003 | `wd1.0_A_s2003.tar.xz` | 160,992,484 | `8683677b221b0fa99bc7de178d0d17925e1214953eb153d27b9ad736980e48e6` | `a1e63a5e-505d-44c7-a3e4-1054ac3880af` | pending owner check |
| wd1.0_A_s2004 | `wd1.0_A_s2004.tar.xz` | 160,809,596 | `3d3583d1a53eb64ad41452c44a80b3e182d45c1e74b8b0b3328a57063cad2408` | `437ad746-d637-4165-8ce0-217c2a32c14f` | pending owner check |
| wd1.0_A_s2005 | `wd1.0_A_s2005.tar.xz` | 160,784,756 | `032a5da5b47482328b5218dbe28a091fd90a815e930c9c293373e59d95a7606c` | `3ca89ab9-e527-40a6-88f1-22435b6cb2a6` | pending owner check |
| wd1.0_A_s2006 | `wd1.0_A_s2006.tar.xz` | 160,278,884 | `5431913267d61dca925268f30f7558e1dd0eb56365ec916cbf8deb8884aff0b9` | `d502aeaf-48cf-46a5-9567-bb12f621ba51` | pending owner check |
| wd1.0_A_s2007 | `wd1.0_A_s2007.tar.xz` | 160,795,840 | `3991421fe3abe3b5c2ac2c2b9ef2f44cc6ca22c21265adcbe8404cc68c12fc33` | `9d395669-6c7a-416b-93b4-d67f04001cdc` | pending owner check |
| wd1.0_A_s2008 | `wd1.0_A_s2008.tar.xz` | 160,559,808 | `591c78d0d0353eab91eee1807286ab0cd2cb2ea6c798501d8d581df646535740` | `90997de5-be8a-4c65-88d6-4f6a5ec29548` | pending owner check |
| wd1.0_A_s2009 | `wd1.0_A_s2009.tar.xz` | 160,157,108 | `8ea814cc6545a493f261c1f9ef752153041184a53fe7d23fc99282d0b4b5d0b8` | `a314f406-fd17-45b9-b862-e8b89f09001e` | pending owner check |
| wd1.0_A_s2010 | `wd1.0_A_s2010.tar.xz` | 160,980,104 | `2c6493ece6e91a2cc11be6f84ac027865529d23e7d8d898e5be1deaf99053d32` | `9fe24104-6ebe-4ba8-bd56-bfb521d4ec26` | pending owner check |
| wd1.0_A_s2011 | `wd1.0_A_s2011.tar.xz` | 160,834,408 | `91124f9b167d1d31dfcf49da359b9ff3deedc476d382d784d30b73c6e42b1d77` | `59248bfc-82c7-47b3-8914-401034151397` | pending owner check |
| wd1.0_A_s2012 | `wd1.0_A_s2012.tar.xz` | 160,888,576 | `eb9a6bf21b9e691cdea53dc67183d5f778b916806cf4b6e9906893f529fe93c8` | `d6551c4f-2ef7-4e6b-817e-d31da1e0a9cd` | pending owner check |
| wd1.0_B_s2001 | `wd1.0_B_s2001.tar.xz` | 161,431,236 | `278baee2fc2fee685e4d2ca55e10fd96d7043f806fd5a11475bc485ee3829aa8` | `73af5ea3-9a02-46e3-95c7-5c57ff3b7699` | pending owner check |
| wd1.0_B_s2002 | `wd1.0_B_s2002.tar.xz` | 161,339,144 | `c316402bd297f58232279cf68fe2373c567da09e6c0648f599a6451b6f6d9cbf` | `fb68b545-d5f4-4e33-8f7d-4c6956084286` | pending owner check |
| wd1.0_B_s2003 | `wd1.0_B_s2003.tar.xz` | 161,386,116 | `de90da38f68c410f88b8c41135450414e3fcd07689b842a1389dba57c559a4ee` | `22bab2dd-2b13-49cb-bb21-bc4f80284b6e` | pending owner check |
| wd1.0_B_s2004 | `wd1.0_B_s2004.tar.xz` | 161,444,180 | `72ac38e766ccf4b8042d10888d9f5ffe578eece4ae420491f4fa2201ed7e4e9b` | `b5827fc3-ce7e-49dd-b08f-24143adbe640` | pending owner check |
| wd1.0_B_s2005 | `wd1.0_B_s2005.tar.xz` | 161,327,804 | `b59bc390caaa25f7d0a7abec22e66d78a541b2ba1f866fd258445a19d6a71915` | `5a1fcaca-5a93-48bb-9400-99bddbce1883` | pending owner check |
| wd1.0_B_s2006 | `wd1.0_B_s2006.tar.xz` | 161,464,112 | `2d9e8b33a6816b7a1728fe966d8ce7bf0a8b21e50c534a3344a573d79151529f` | `10e36702-e414-441c-8f9d-b840dd684300` | pending owner check |
| wd1.0_B_s2007 | `wd1.0_B_s2007.tar.xz` | 161,402,464 | `e039d16086383e1be664dbbd72ade4298fe6ab097174be8ca49e7172643eaa53` | `bade6059-2d15-4d89-a30e-f499ca6cdb5f` | pending owner check |
| wd1.0_B_s2008 | `wd1.0_B_s2008.tar.xz` | 161,320,656 | `b0a6c6ad6964a6981ddf0b7bd6b115234aa251b88d887fe47f7c0f71acb092de` | `ff02fd0d-1599-49e9-8117-a7f9b7ea7939` | pending owner check |
| wd1.0_B_s2009 | `wd1.0_B_s2009.tar.xz` | 161,374,180 | `4e9a76c6bba5df102fb3f087b543b2cc2fd9ea4b437217761d618ac3be6417ba` | `4a852f0f-ffc3-46c7-b4c6-3f36631080b1` | pending owner check |
| wd1.0_B_s2010 | `wd1.0_B_s2010.tar.xz` | 161,434,940 | `07a0ebc1946ceb9b081830d8aaf16cff0344c74c7cf86fec0f31f0a0a603584e` | `bcca22e4-71ba-43ab-ab5e-ae98eb3fcab5` | pending owner check |
| wd1.0_B_s2011 | `wd1.0_B_s2011.tar.xz` | 161,612,520 | `019b8300419fe39df8de9da180b52116ea92d2ffbab5c01c9ebc77e2ae8f5697` | `39277ea3-f20f-4f6b-8610-416654df0d16` | pending owner check |
| wd1.0_B_s2012 | `wd1.0_B_s2012.tar.xz` | 161,516,672 | `3504325a9ff023040320df78a6d9e9ba8ba117565d021f92a0f462461a600a37` | `5ce4c557-031d-4224-aeac-19b22e36daea` | pending owner check |
| wd1.0_C_s2001 | `wd1.0_C_s2001.tar.xz` | 162,657,576 | `217a0f407301243455054f973ca6d1f6f01c157020f9f690b5d1489e999840ae` | `8883345e-bc4e-4689-a57c-4b6bd605878e` | pending owner check |
| wd1.0_C_s2002 | `wd1.0_C_s2002.tar.xz` | 162,672,872 | `7c14644649234fbe05320bd4681e800e0b7dfd49502227c0f3a2b0a5a90646a2` | `211de052-04c2-46fd-a9fa-3b230ae25bc6` | pending owner check |
| wd1.0_C_s2003 | `wd1.0_C_s2003.tar.xz` | 162,658,604 | `4f77d492cc721b42fad76084075c4e699514f9c798906dc48c2b3721785ad046` | `1eca00d7-9915-4831-94f2-1e7764de8d09` | pending owner check |
| wd1.0_C_s2004 | `wd1.0_C_s2004.tar.xz` | 162,739,940 | `53df2e7f6920b17078fbafe4fc079f9295ff81e69189466e92d16c3cfeaad3fa` | `a9c14ad4-1c52-40d0-afb8-78085a8deb73` | pending owner check |
| wd1.0_C_s2005 | `wd1.0_C_s2005.tar.xz` | 162,551,664 | `f015cd329f280e5f28eb9cef29cf3108769b53c1fb83572555268b8f11ae0391` | `f17cef6a-639a-4596-8c1c-337b5636ec6d` | pending owner check |
| wd1.0_C_s2006 | `wd1.0_C_s2006.tar.xz` | 162,745,968 | `eb73f3646bd039530929a5e52022cdcd878b33464748103e5a553d669eeebff9` | `2a7c7325-456e-48a7-81d0-7973ec278790` | pending owner check |
| wd1.0_C_s2007 | `wd1.0_C_s2007.tar.xz` | 162,708,276 | `28884bbf2aa148a468a51381fbabb8eecf3dec08a59305b288ce471823b57420` | `b67e88d5-a0a5-4ea7-8239-0c5710b18595` | pending owner check |
| wd1.0_C_s2008 | `wd1.0_C_s2008.tar.xz` | 162,725,112 | `f30f1086cebc368ded1057bb227f1cd45babe84e8d647b01efbbcca4c81e0706` | `8414d50e-bcc2-4d74-a287-b46fa035f481` | pending owner check |
| wd1.0_C_s2009 | `wd1.0_C_s2009.tar.xz` | 162,352,796 | `54536076e0807b6fe035a44cbfc08126e797da46ecd39efc7a30ba1b8be07544` | `ffcb1128-f6c9-4605-84b7-976ae459e1f6` | pending owner check |
| wd1.0_C_s2010 | `wd1.0_C_s2010.tar.xz` | 162,549,588 | `924077949b0f90ee7d946896e4cf72e9882ad34b93d7c82a9e552efa23f26372` | `cac0d9f2-368e-4eec-ae91-b995a25b0aa9` | pending owner check |
| wd1.0_C_s2011 | `wd1.0_C_s2011.tar.xz` | 162,823,196 | `c896d6070579192d7a90a14d248cb5b930417445b2739b70f8ccf3d79c946664` | `e59aa42f-dfd8-4cd5-a24b-f668f76c89fe` | pending owner check |
| wd1.0_C_s2012 | `wd1.0_C_s2012.tar.xz` | 162,652,404 | `46bd759c84ffd4fe8cd93cf6105cbd17e679abe59182802a202300e49ed61a05` | `bc7c1210-734f-494f-a0e4-4ce8746ee4eb` | pending owner check |
| wd1.0_D_s2001 | `wd1.0_D_s2001.tar.xz` | 162,615,012 | `13fb3fc8c9905795fad7ea741db6ba6255648f70fdf53941a117daba86bac648` | `f4cb750d-a963-4401-8643-27a3ce383643` | pending owner check |
| wd1.0_D_s2002 | `wd1.0_D_s2002.tar.xz` | 162,593,808 | `1adafd73300941119785697f73b5bc0683e9705739a041b9ea813159bf95eca8` | `180e1d75-c8de-45d2-b9fb-b99a04fac3af` | pending owner check |
| wd1.0_D_s2003 | `wd1.0_D_s2003.tar.xz` | 162,422,872 | `2f3bfbb305f11a423191add6e3f3176e1230db0aacd80590a2cc44c306269ee8` | `d374bd52-4596-429b-80c0-eef7360ded52` | pending owner check |
| wd1.0_D_s2004 | `wd1.0_D_s2004.tar.xz` | 162,690,336 | `f4e4cc8da45de82e4b55462c9b31849c952b24eb11c9ad907132c8c2035a0e42` | `051c3fe5-a9c4-4d4e-befe-01ec8e529658` | pending owner check |
| wd1.0_D_s2005 | `wd1.0_D_s2005.tar.xz` | 162,443,204 | `ce991a52e7abb922736bec2bda6e042a4eead74d4079ac2b9be16236bf301680` | `7813d144-405d-4e7c-bf8d-99c7b83856ce` | pending owner check |
| wd1.0_D_s2006 | `wd1.0_D_s2006.tar.xz` | 162,505,728 | `d21e0e4af663ab5eb65aa4dcba5962cdf35158a223f4c32c10bb318f1928757e` | `44a60df0-d74b-44b2-8626-52e534e00594` | pending owner check |
| wd1.0_D_s2007 | `wd1.0_D_s2007.tar.xz` | 162,552,380 | `98f982adf7f35dd52afbf90191b5c22cb3cc40c16a668c1c44e64ff5d4332f21` | `696f8aa1-1192-40a5-a612-03b04db88369` | pending owner check |
| wd1.0_D_s2008 | `wd1.0_D_s2008.tar.xz` | 162,724,144 | `db87a36be03bc66ab4b9453709069461baea81351c53a56e6d4ce9b3f5b93217` | `38410336-8435-45c6-abf7-e8fae05341bd` | pending owner check |
| wd1.0_D_s2009 | `wd1.0_D_s2009.tar.xz` | 162,621,972 | `d5bccbc8adcd222f379784e61c66489697271f09a751ca580b327b0d86f89779` | `f2f9b8ce-d084-4a48-8c68-b8c64d0cc15b` | pending owner check |
| wd1.0_D_s2010 | `wd1.0_D_s2010.tar.xz` | 162,467,876 | `cc61f7c2a715c48ab3046973edd09cf48ac8942d26698bdd92dfa5c92baa6fe8` | `15044330-55d2-4fce-9194-0e7ed6c1140f` | pending owner check |
| wd1.0_D_s2011 | `wd1.0_D_s2011.tar.xz` | 162,656,032 | `09f38bbd7133ceecee55d8a171392ec55020e0f1b8530b84b91c2e6bdb476eb0` | `7bd352ba-2677-4054-977a-3e0aad78c860` | pending owner check |
| wd1.0_D_s2012 | `wd1.0_D_s2012.tar.xz` | 162,696,132 | `ace830051f3fa82204f31f782e8a4d5c8eee47e3baaca4f3f4c345d761bedd48` | `2bb27fdc-f4ac-4b77-9ed9-2381201baeee` | pending owner check |
