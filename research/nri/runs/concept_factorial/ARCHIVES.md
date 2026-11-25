# Checkpoint archives — concept factorial (96 runs)

Complete 100-step checkpoints (61 per run, step 0-6,000: weights, auxiliary head, AdamW state, torch + Python RNG,
AdamW/weight-decay update decomposition, per-checkpoint audit tensors) plus the per-step interval log, one `tar.xz`
per run. Each archive was verified byte-for-byte against the live files with `tar --diff` before the live copy was
removed; every file's SHA-256 is in `<run>/manifest.json`.

**Location at creation: `/home/user/nri_archives/` inside the Claude Code session container (≈ 15 GB). That container
is ephemeral: copy the archives to durable storage to preserve them.** Verify a copy with
`sha256sum <run>.tar.xz` against the table, and the contents with `tar -xJf <run>.tar.xz` + the manifest hashes.

## Regenerating instead of restoring

Training is deterministic (`torch.use_deterministic_algorithms`, seeded data streams, round-robin scheduling
tested equal to standalone runs). Verified in the original environment: re-running `wd1.0_C_s2004` from the
committed code for 300 steps reproduced `model.safetensors`, `optim.safetensors` and `updates.safetensors` at
steps 0, 100, 200 and 300 **byte-identically** (SHA-256 equal to this manifest). Any run can therefore be
regenerated and checked file-by-file against its manifest:

```
python - <<'PY'
from pathlib import Path
from sci import factorial as Fx
Fx.train_worker(0, 1, live=Path("regen"), runs=[("C", 1.0, 2004)], steps=6000)   # ~10 CPU-min per network
PY
# then compare sha256 of regen/<run>/ckpt/... with runs/concept_factorial/<run>/manifest.json
```

Caveat: bit-identity is only established for the same software and CPU class (Python 3.13, the session's PyTorch
build, x86-64); a different PyTorch version or CPU can change floating-point summation order. The manifests make
any such divergence detectable. Audit tensors are recomputed with `python -m sci.factorial audit`.

| run | archive | bytes | sha256 | files |
|---|---|---|---|---|
| wd0.1_A_s2001 | `wd0.1_A_s2001.tar.xz` | 159,920,452 | `4daae94496848b1d728f54689f0e78b2112493f4eb4e7fbeebb5eb462e2e7e07` | 612 |
| wd0.1_A_s2002 | `wd0.1_A_s2002.tar.xz` | 159,868,196 | `81579129cf178e8fb3c157d4ed3f20fd6a0734c8daa48632ff3f849766ea04bc` | 612 |
| wd0.1_A_s2003 | `wd0.1_A_s2003.tar.xz` | 160,111,484 | `2691627b6da418169e7a0e860dd669289b8a80a3bf3e90a0158c7c53f53eb9d6` | 612 |
| wd0.1_A_s2004 | `wd0.1_A_s2004.tar.xz` | 160,089,348 | `2535136e6c1f201b61fbd1b3e246c0b9d2add56bb468d89966ec424ab4112ac2` | 612 |
| wd0.1_A_s2005 | `wd0.1_A_s2005.tar.xz` | 160,102,904 | `200f0d1819fac8aa94094e1bec66f7b5573f9a55f0be11d48f0480cd9b7bc177` | 612 |
| wd0.1_A_s2006 | `wd0.1_A_s2006.tar.xz` | 160,156,860 | `80b08a759231bb746a8e63a3f0a9106ae3c77973a84b7fbc328efc1f93047f73` | 612 |
| wd0.1_A_s2007 | `wd0.1_A_s2007.tar.xz` | 160,079,780 | `4023cf812b107583179be277cbe12a1252584dc0367b74fe86f051a08e0dc035` | 612 |
| wd0.1_A_s2008 | `wd0.1_A_s2008.tar.xz` | 159,937,060 | `7008c9adb20ae0d9b3245ecdd55bdb37284df0d36cf4d70bcb76f854ff11e41d` | 612 |
| wd0.1_A_s2009 | `wd0.1_A_s2009.tar.xz` | 160,127,588 | `6b49f641e9ad491b8cce4fe1b71e6c504963df25b804e60abbf25fb521972e7b` | 612 |
| wd0.1_A_s2010 | `wd0.1_A_s2010.tar.xz` | 160,138,172 | `3faec0da819add67dedd7ac10e82ad3afb47843184edf10971b00a39bd5b9c10` | 612 |
| wd0.1_A_s2011 | `wd0.1_A_s2011.tar.xz` | 160,080,820 | `f8e7e94ae2f30b2e74236ab431609262eefa053ef2e7234aa65df781e54066f2` | 612 |
| wd0.1_A_s2012 | `wd0.1_A_s2012.tar.xz` | 159,858,788 | `5397eb3526394a09b9bc29f509db6318a97277b46c4c2e75ad178d11ea75b7db` | 612 |
| wd0.1_B_s2001 | `wd0.1_B_s2001.tar.xz` | 161,245,444 | `556f723d69d6b8bd33ae67046e5e02601deda4d26e49c872ecfef486ed32e768` | 612 |
| wd0.1_B_s2002 | `wd0.1_B_s2002.tar.xz` | 161,138,904 | `0b8455507e73f118f7d43e0148bacab8d25933fe824c4191e54214fc46cdb674` | 612 |
| wd0.1_B_s2003 | `wd0.1_B_s2003.tar.xz` | 161,097,484 | `3b99adb2fa4eea163c0c9bf725923f90c7d4f2bd608c811bfcf40c570a87358a` | 612 |
| wd0.1_B_s2004 | `wd0.1_B_s2004.tar.xz` | 161,124,016 | `9bcd273f8eb1e4f3de41e41ed314bf687f1078958b63178e7f58b8283b143e23` | 612 |
| wd0.1_B_s2005 | `wd0.1_B_s2005.tar.xz` | 161,157,292 | `f5540f4e85b4055027aa0d2b1c30446057ccd730d6c9c9292edd152ba1ce86fb` | 612 |
| wd0.1_B_s2006 | `wd0.1_B_s2006.tar.xz` | 161,220,516 | `93379d5a2a3aa6ff5b917b970feb6ed0a687223fd5e55ebf667957e55982f131` | 612 |
| wd0.1_B_s2007 | `wd0.1_B_s2007.tar.xz` | 161,042,724 | `975f383b051fdf90ad00dc704eb57c990bb6bb0bbfcc3c8fd7da12bd5574837b` | 612 |
| wd0.1_B_s2008 | `wd0.1_B_s2008.tar.xz` | 161,174,548 | `08494b2834ad49171b22035e31b65b8cc168264894e5ae1b8c184e5ab9772d3f` | 612 |
| wd0.1_B_s2009 | `wd0.1_B_s2009.tar.xz` | 161,167,900 | `84249aa5e1976826b9339ce4c0eecbe8ddbd3989041d70c7f2d73ddebb5bd53a` | 612 |
| wd0.1_B_s2010 | `wd0.1_B_s2010.tar.xz` | 161,211,208 | `8e7785a7dc8c3dc64815295af3f37367801b14298383539ecac84bdfb83f410a` | 612 |
| wd0.1_B_s2011 | `wd0.1_B_s2011.tar.xz` | 161,310,288 | `80f0e9652f0b90bafe73c82ce8879f10e0b9bbe046ea9c7422d3dea06f6d9d8e` | 612 |
| wd0.1_B_s2012 | `wd0.1_B_s2012.tar.xz` | 161,165,052 | `978fd228fff3edcc0fe6fd2c92e62e9911d9cc6fafa26bbcf836db4aae4245df` | 612 |
| wd0.1_C_s2001 | `wd0.1_C_s2001.tar.xz` | 162,374,348 | `91e0edfd8a497ebfc4b34e6c552c74636b073be20ed706b6b2c0ac2f71042afa` | 673 |
| wd0.1_C_s2002 | `wd0.1_C_s2002.tar.xz` | 162,516,484 | `82a70fef29477874e3fa8f2ac19df1649fe4ed5cb8a8d8b2d47647656dced315` | 673 |
| wd0.1_C_s2003 | `wd0.1_C_s2003.tar.xz` | 162,368,940 | `fbd2cfe124b3f53f1a9a6295832d5154ae88fecb24d029f3e7aa112870a9f77c` | 673 |
| wd0.1_C_s2004 | `wd0.1_C_s2004.tar.xz` | 162,497,716 | `9fa436e975eac496a4457ac1b380b4210c2a7ad2a229439e611318ccd57d0d84` | 673 |
| wd0.1_C_s2005 | `wd0.1_C_s2005.tar.xz` | 162,458,296 | `44acd534fe0c321e58f83f7bbf2b160c0236ebb379d8f117aa36a82ce3eed4c7` | 673 |
| wd0.1_C_s2006 | `wd0.1_C_s2006.tar.xz` | 162,527,088 | `a96c1308d64c9d6569d1ccce2968d460da0c5bdf7794410bd9e0ed6cabadd4fd` | 673 |
| wd0.1_C_s2007 | `wd0.1_C_s2007.tar.xz` | 162,451,004 | `5cb3e23f3d3fd5e1b621044a6dba4760fb156fd3ef2f07dbfdcb25be71d1969f` | 673 |
| wd0.1_C_s2008 | `wd0.1_C_s2008.tar.xz` | 162,476,040 | `696f2c3af9f7f8a856f467f0834e8b7f402b1a83074ea13754ac50ddc09dc602` | 673 |
| wd0.1_C_s2009 | `wd0.1_C_s2009.tar.xz` | 162,482,420 | `0f717490abc224d6f20834cf40ec0c71e2207a280b3dc328efe0c9fbddff9be0` | 673 |
| wd0.1_C_s2010 | `wd0.1_C_s2010.tar.xz` | 162,467,236 | `3b115208a09f34a001231f3ddcb89be7099e2e9686551a0c99481606dfa7f22f` | 673 |
| wd0.1_C_s2011 | `wd0.1_C_s2011.tar.xz` | 162,648,940 | `d0d2455b0a467d11f283106059b38b7ed7714b8b9df1a39217f80057844de38e` | 673 |
| wd0.1_C_s2012 | `wd0.1_C_s2012.tar.xz` | 162,661,120 | `ce3f67f4b50f110955d2fb574fc8962410f42c380522d401337139382a5806b2` | 673 |
| wd0.1_D_s2001 | `wd0.1_D_s2001.tar.xz` | 162,394,888 | `304e98d0b4e6d1800dc7d27009af01c1d5afc2184056ef3428a3671f50ff6927` | 673 |
| wd0.1_D_s2002 | `wd0.1_D_s2002.tar.xz` | 162,469,132 | `185b20527d94212e5456aa73e0205d29d932aff8dd7b72461af2697b1fbb716c` | 673 |
| wd0.1_D_s2003 | `wd0.1_D_s2003.tar.xz` | 162,383,304 | `3fa83eff5de64fbd6e54a6af3f645e004d76ce3378ea4d96e372c0f8e237cf30` | 673 |
| wd0.1_D_s2004 | `wd0.1_D_s2004.tar.xz` | 162,381,324 | `61cf1087918df937f1e9523c56b902c5feeb4a9404547d4485c914b3681bc792` | 673 |
| wd0.1_D_s2005 | `wd0.1_D_s2005.tar.xz` | 162,371,744 | `d85052a22fef08880a3caa07f9964ffc6a1d9ee6bdd1d9fc1fcfb3eab4eb1489` | 673 |
| wd0.1_D_s2006 | `wd0.1_D_s2006.tar.xz` | 162,374,312 | `f9fe72cdfef484c45d4b3c483c0b28938527d4f54036a9850af387c35fa74f1e` | 673 |
| wd0.1_D_s2007 | `wd0.1_D_s2007.tar.xz` | 162,396,124 | `5d8fecb2bd79c2eb36ed2e4e15af9f27a0c43e401be16f8f4206ffeb7603cc1b` | 673 |
| wd0.1_D_s2008 | `wd0.1_D_s2008.tar.xz` | 162,447,596 | `9af41b4f8d99e6cdf79f9de008b48b7264b06ee37129b46f62c1fabf9c63c1ee` | 673 |
| wd0.1_D_s2009 | `wd0.1_D_s2009.tar.xz` | 162,492,744 | `8c433009480033a6538f9143a4232eb8844de72f0e9e2b4473dfb4afe7ac2a6a` | 673 |
| wd0.1_D_s2010 | `wd0.1_D_s2010.tar.xz` | 162,435,348 | `90a878c29ff7e07e783746e170c0a4553e5225d2485d0ea34be8210bbc40d08c` | 673 |
| wd0.1_D_s2011 | `wd0.1_D_s2011.tar.xz` | 162,554,524 | `03e4ca1297dde504721a7688d28cb4ce2bd046e2c83320727ba85cbe9751f7dd` | 673 |
| wd0.1_D_s2012 | `wd0.1_D_s2012.tar.xz` | 162,430,748 | `2833259ed6619f086b52fd0aeafc876ccd3b88f1022fb223be1965c7af839012` | 673 |
| wd1.0_A_s2001 | `wd1.0_A_s2001.tar.xz` | 160,686,312 | `580e51bb7f2af26e54af576183f711824605eb92747b0e729921cb184314440e` | 612 |
| wd1.0_A_s2002 | `wd1.0_A_s2002.tar.xz` | 160,736,416 | `71dc9730c1d2edc9c1b23f0e886c6f84df9c1cf8eed9e49401ff25a978a5d16f` | 612 |
| wd1.0_A_s2003 | `wd1.0_A_s2003.tar.xz` | 160,992,484 | `8683677b221b0fa99bc7de178d0d17925e1214953eb153d27b9ad736980e48e6` | 612 |
| wd1.0_A_s2004 | `wd1.0_A_s2004.tar.xz` | 160,809,596 | `3d3583d1a53eb64ad41452c44a80b3e182d45c1e74b8b0b3328a57063cad2408` | 612 |
| wd1.0_A_s2005 | `wd1.0_A_s2005.tar.xz` | 160,784,756 | `032a5da5b47482328b5218dbe28a091fd90a815e930c9c293373e59d95a7606c` | 612 |
| wd1.0_A_s2006 | `wd1.0_A_s2006.tar.xz` | 160,278,884 | `5431913267d61dca925268f30f7558e1dd0eb56365ec916cbf8deb8884aff0b9` | 612 |
| wd1.0_A_s2007 | `wd1.0_A_s2007.tar.xz` | 160,795,840 | `3991421fe3abe3b5c2ac2c2b9ef2f44cc6ca22c21265adcbe8404cc68c12fc33` | 612 |
| wd1.0_A_s2008 | `wd1.0_A_s2008.tar.xz` | 160,559,808 | `591c78d0d0353eab91eee1807286ab0cd2cb2ea6c798501d8d581df646535740` | 612 |
| wd1.0_A_s2009 | `wd1.0_A_s2009.tar.xz` | 160,157,108 | `8ea814cc6545a493f261c1f9ef752153041184a53fe7d23fc99282d0b4b5d0b8` | 612 |
| wd1.0_A_s2010 | `wd1.0_A_s2010.tar.xz` | 160,980,104 | `2c6493ece6e91a2cc11be6f84ac027865529d23e7d8d898e5be1deaf99053d32` | 612 |
| wd1.0_A_s2011 | `wd1.0_A_s2011.tar.xz` | 160,834,408 | `91124f9b167d1d31dfcf49da359b9ff3deedc476d382d784d30b73c6e42b1d77` | 612 |
| wd1.0_A_s2012 | `wd1.0_A_s2012.tar.xz` | 160,888,576 | `eb9a6bf21b9e691cdea53dc67183d5f778b916806cf4b6e9906893f529fe93c8` | 612 |
| wd1.0_B_s2001 | `wd1.0_B_s2001.tar.xz` | 161,431,236 | `278baee2fc2fee685e4d2ca55e10fd96d7043f806fd5a11475bc485ee3829aa8` | 612 |
| wd1.0_B_s2002 | `wd1.0_B_s2002.tar.xz` | 161,339,144 | `c316402bd297f58232279cf68fe2373c567da09e6c0648f599a6451b6f6d9cbf` | 612 |
| wd1.0_B_s2003 | `wd1.0_B_s2003.tar.xz` | 161,386,116 | `de90da38f68c410f88b8c41135450414e3fcd07689b842a1389dba57c559a4ee` | 612 |
| wd1.0_B_s2004 | `wd1.0_B_s2004.tar.xz` | 161,444,180 | `72ac38e766ccf4b8042d10888d9f5ffe578eece4ae420491f4fa2201ed7e4e9b` | 612 |
| wd1.0_B_s2005 | `wd1.0_B_s2005.tar.xz` | 161,327,804 | `b59bc390caaa25f7d0a7abec22e66d78a541b2ba1f866fd258445a19d6a71915` | 612 |
| wd1.0_B_s2006 | `wd1.0_B_s2006.tar.xz` | 161,464,112 | `2d9e8b33a6816b7a1728fe966d8ce7bf0a8b21e50c534a3344a573d79151529f` | 612 |
| wd1.0_B_s2007 | `wd1.0_B_s2007.tar.xz` | 161,402,464 | `e039d16086383e1be664dbbd72ade4298fe6ab097174be8ca49e7172643eaa53` | 612 |
| wd1.0_B_s2008 | `wd1.0_B_s2008.tar.xz` | 161,320,656 | `b0a6c6ad6964a6981ddf0b7bd6b115234aa251b88d887fe47f7c0f71acb092de` | 612 |
| wd1.0_B_s2009 | `wd1.0_B_s2009.tar.xz` | 161,374,180 | `4e9a76c6bba5df102fb3f087b543b2cc2fd9ea4b437217761d618ac3be6417ba` | 612 |
| wd1.0_B_s2010 | `wd1.0_B_s2010.tar.xz` | 161,434,940 | `07a0ebc1946ceb9b081830d8aaf16cff0344c74c7cf86fec0f31f0a0a603584e` | 612 |
| wd1.0_B_s2011 | `wd1.0_B_s2011.tar.xz` | 161,612,520 | `019b8300419fe39df8de9da180b52116ea92d2ffbab5c01c9ebc77e2ae8f5697` | 612 |
| wd1.0_B_s2012 | `wd1.0_B_s2012.tar.xz` | 161,516,672 | `3504325a9ff023040320df78a6d9e9ba8ba117565d021f92a0f462461a600a37` | 612 |
| wd1.0_C_s2001 | `wd1.0_C_s2001.tar.xz` | 162,657,576 | `217a0f407301243455054f973ca6d1f6f01c157020f9f690b5d1489e999840ae` | 673 |
| wd1.0_C_s2002 | `wd1.0_C_s2002.tar.xz` | 162,672,872 | `7c14644649234fbe05320bd4681e800e0b7dfd49502227c0f3a2b0a5a90646a2` | 673 |
| wd1.0_C_s2003 | `wd1.0_C_s2003.tar.xz` | 162,658,604 | `4f77d492cc721b42fad76084075c4e699514f9c798906dc48c2b3721785ad046` | 673 |
| wd1.0_C_s2004 | `wd1.0_C_s2004.tar.xz` | 162,739,940 | `53df2e7f6920b17078fbafe4fc079f9295ff81e69189466e92d16c3cfeaad3fa` | 673 |
| wd1.0_C_s2005 | `wd1.0_C_s2005.tar.xz` | 162,551,664 | `f015cd329f280e5f28eb9cef29cf3108769b53c1fb83572555268b8f11ae0391` | 673 |
| wd1.0_C_s2006 | `wd1.0_C_s2006.tar.xz` | 162,745,968 | `eb73f3646bd039530929a5e52022cdcd878b33464748103e5a553d669eeebff9` | 673 |
| wd1.0_C_s2007 | `wd1.0_C_s2007.tar.xz` | 162,708,276 | `28884bbf2aa148a468a51381fbabb8eecf3dec08a59305b288ce471823b57420` | 673 |
| wd1.0_C_s2008 | `wd1.0_C_s2008.tar.xz` | 162,725,112 | `f30f1086cebc368ded1057bb227f1cd45babe84e8d647b01efbbcca4c81e0706` | 673 |
| wd1.0_C_s2009 | `wd1.0_C_s2009.tar.xz` | 162,352,796 | `54536076e0807b6fe035a44cbfc08126e797da46ecd39efc7a30ba1b8be07544` | 673 |
| wd1.0_C_s2010 | `wd1.0_C_s2010.tar.xz` | 162,549,588 | `924077949b0f90ee7d946896e4cf72e9882ad34b93d7c82a9e552efa23f26372` | 673 |
| wd1.0_C_s2011 | `wd1.0_C_s2011.tar.xz` | 162,823,196 | `c896d6070579192d7a90a14d248cb5b930417445b2739b70f8ccf3d79c946664` | 673 |
| wd1.0_C_s2012 | `wd1.0_C_s2012.tar.xz` | 162,652,404 | `46bd759c84ffd4fe8cd93cf6105cbd17e679abe59182802a202300e49ed61a05` | 673 |
| wd1.0_D_s2001 | `wd1.0_D_s2001.tar.xz` | 162,615,012 | `13fb3fc8c9905795fad7ea741db6ba6255648f70fdf53941a117daba86bac648` | 673 |
| wd1.0_D_s2002 | `wd1.0_D_s2002.tar.xz` | 162,593,808 | `1adafd73300941119785697f73b5bc0683e9705739a041b9ea813159bf95eca8` | 673 |
| wd1.0_D_s2003 | `wd1.0_D_s2003.tar.xz` | 162,422,872 | `2f3bfbb305f11a423191add6e3f3176e1230db0aacd80590a2cc44c306269ee8` | 673 |
| wd1.0_D_s2004 | `wd1.0_D_s2004.tar.xz` | 162,690,336 | `f4e4cc8da45de82e4b55462c9b31849c952b24eb11c9ad907132c8c2035a0e42` | 673 |
| wd1.0_D_s2005 | `wd1.0_D_s2005.tar.xz` | 162,443,204 | `ce991a52e7abb922736bec2bda6e042a4eead74d4079ac2b9be16236bf301680` | 673 |
| wd1.0_D_s2006 | `wd1.0_D_s2006.tar.xz` | 162,505,728 | `d21e0e4af663ab5eb65aa4dcba5962cdf35158a223f4c32c10bb318f1928757e` | 673 |
| wd1.0_D_s2007 | `wd1.0_D_s2007.tar.xz` | 162,552,380 | `98f982adf7f35dd52afbf90191b5c22cb3cc40c16a668c1c44e64ff5d4332f21` | 673 |
| wd1.0_D_s2008 | `wd1.0_D_s2008.tar.xz` | 162,724,144 | `db87a36be03bc66ab4b9453709069461baea81351c53a56e6d4ce9b3f5b93217` | 673 |
| wd1.0_D_s2009 | `wd1.0_D_s2009.tar.xz` | 162,621,972 | `d5bccbc8adcd222f379784e61c66489697271f09a751ca580b327b0d86f89779` | 673 |
| wd1.0_D_s2010 | `wd1.0_D_s2010.tar.xz` | 162,467,876 | `cc61f7c2a715c48ab3046973edd09cf48ac8942d26698bdd92dfa5c92baa6fe8` | 673 |
| wd1.0_D_s2011 | `wd1.0_D_s2011.tar.xz` | 162,656,032 | `09f38bbd7133ceecee55d8a171392ec55020e0f1b8530b84b91c2e6bdb476eb0` | 673 |
| wd1.0_D_s2012 | `wd1.0_D_s2012.tar.xz` | 162,696,132 | `ace830051f3fa82204f31f782e8a4d5c8eee47e3baaca4f3f4c345d761bedd48` | 673 |
