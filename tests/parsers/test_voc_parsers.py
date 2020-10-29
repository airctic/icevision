from icevision.all import *


def test_voc_annotation_parser(samples_source, voc_class_map):
    annotation_parser = parsers.voc(
        annotations_dir=samples_source / "voc/Annotations",
        images_dir=samples_source / "voc/JPEGImages",
        class_map=voc_class_map,
    )
    records = annotation_parser.parse(data_splitter=SingleSplitSplitter())[0]

    assert len(records) == 2

    record = records[0]
    expected = {
        "imageid": 0,
        "filepath": samples_source / "voc/JPEGImages/2007_000063.jpg",
        "width": 500,
        "height": 375,
        "labels": [voc_class_map.get_name(k) for k in ["dog", "chair"]],
        "bboxes": [BBox.from_xyxy(123, 115, 379, 275), BBox.from_xyxy(75, 1, 428, 375)],
    }
    assert record == expected

    record = records[1]
    expected = {
        "imageid": 1,
        "filepath": samples_source / "voc/JPEGImages/2011_003353.jpg",
        "height": 500,
        "width": 375,
        "labels": [voc_class_map.get_name("person")],
        "bboxes": [BBox.from_xyxy(130, 45, 375, 470)],
    }
    assert record == expected


def test_voc_mask_parser(samples_source, voc_class_map):
    parser = parsers.voc(
        annotations_dir=samples_source / "voc/Annotations",
        images_dir=samples_source / "voc/JPEGImages",
        class_map=voc_class_map,
        masks_dir=samples_source / "voc/SegmentationClass",
    )

    records = parser.parse(data_splitter=SingleSplitSplitter())[0]

    assert len(records) == 1

    record = records[0]
    expected = {
        "imageid": 0,
        "filepath": samples_source / "voc/JPEGImages/2007_000063.jpg",
        "width": 500,
        "height": 375,
        "labels": [voc_class_map.get_name(k) for k in ["dog", "chair"]],
        "bboxes": [BBox.from_xyxy(123, 115, 379, 275), BBox.from_xyxy(75, 1, 428, 375)],
        "masks": EncodedRLEs(
            [
                {
                    "size": [375, 500],
                    "counts": b"Pjl0h0m9T1cHUNl3j2PL[MP3d3mL_Ll2h3mL`Lm2e3PM^Lj2g3VMYLf2l3XMULd2o3[MRLa2R4^MoK]2V4cMkKX2Y4fMiKV2[4hMgKT2]4iMfKS2^4hMgKS2^4lMcKP2a4PN`Kk1d4UN[Kg1j4XNTKg1P5YNoJc1V5]NiJ`1[5_NfJa1Z5_NeJb1[5^NeJb1[5^NdJc1\\5]NdJc1\\5XMoIc0d0V2]5RMZJb09\\2X6aMaIf2a6XM_Ih2d6UM\\Ik2e6TM[Il2f6SMZIm2g6RMYIn2h6QMXIo2i6PMXIo2i6PMWIP3j6oLVIQ3k6nLUIR3l6mLTIS3m6lLSIT3n6kLRIU3j2oLV1k2kN[MT1l0e0UOZO:W1GhN9X1HgN8Y1IfN7Z1JeN6Y1MfN3W11hNOW13iNLV16iNJW17hNIW19hNGW1;hNEX1<gNDX1>gNBY1?fNAY1a0fN_OY1c0fN]OZ1d0eN\\OZ1f0eNZO[1g0dNYO[1h0eNXOZ1j0eNVO[1k0dNUO[1l0eNTOZ1n0eNRO[1o0dNQO[1Q1eNnN[1S1dNmN[1U1dNkN[1W1dNiN\\1W1RKVNU3c0h1Y1PK`Nn27Q2[1nJcNn22S2W2jMiMU2Z2iMfMV2\\2iMdMV2_2hMaMV2b2iM^MV2e2hM[MW2g2hMYMW2h2iMXMV2i2jMWMT2k2lMUMR2n2mMRMR2o2nMQMQ2P3nMQMR2o2mMRMR2P3lMQMn1\\O`Ke3a2PMm1V3RNkLm1V3RNkLm1V3RNkLm1W3QNjLm1X3TNgLk1Z3UNfLj1[3VNeLi1\\3WNdLh1]3XNcLg1^3ZNaLf1_3ZNaLe1`3[N`Ld1a3\\N_Ld1a3\\N_Lc1b3]N^Lc1a3_N^L`1b3aN^L^1b3cN^L]1b3cN^L\\1b3eN^LZ1b3gN^LY1a3hN_LW1a3kN^LU1a3lN_LS1`3oN`LP1_3ROaLn0^3SObLl0]3VOcLj0[3YOdLg0Z3[OfLe0R3CnL<P3GPM9n2IRM7l2KTM4m2LSM4n2LQM4P3KPM4R3KnL5S3JmL6S3_LXLT3e0<T3_LZLS3b0>T3^L\\LS3`0?T3^L]LS3>?U3]L_LS3<?V3^L^LS3<?V3^L_LR3;`0W3]L_LR3:`0X3^L_LQ36oL_Ob3m3]L_LQ33RMA`3m3]L_LQ31TMC]3]4^OnKWME[3]4^OmKXMFZ3]4^OkKZMHX3^4]OhK]MJV3^4]OfK_MLT3^4^OcK`MOR3^41bKO_40aK0_40aK0_40aK0_40aK0a4N_K2a4N_K2a4N_K2b4M^K3b4M^K3b4M^K2c4N]K2d4M\\K3d4M\\K3d4M\\K3d4M\\K3e4L[K4e4L[K4e4L[K4e4L[K4f4KZK5f4J[K6e4I\\K7d4H]K8c4G^K9c4E^K;b4D_K<a4C`K=`4D_K<b4C^K=b4D]K<c4D]K<c4D]K<c4E\\K<c4D]K<d4C\\K=d4D[K<e4D[K<e4EZK;f4EZK;f4EZK;f4FYK:S4^L_LW3^O;P4aLbLT3^O;P4aLbLT3^O;o3bLcLS3^O;o3bLcLS3^O;n3cLdLR3^O;m3dLfLP3]O=l3cLgLP3]O=k3dLhLo2]O=k3dLhLo2]O=j3eLiLn2]O=j3eLiLn2]O>i3dLkLm2\\O?j3cLjLo2[O>l3bLiLP3[O>l3bLiLP3[O>m3aLhLQ3[O>n3`LgLR3[O?n3^LhLR3ZO`0n3_LgLQ3[O`0o3^LfLR3[O`0P4]LeLS3[O`0P4]LeLS3[Oa0P4[LfLS3ZOb0Q4ZLeLT3ZOb0Q4ZLeLT3ZOb0i4^OWKb0i4^OWKb0j4]OVKd0i4\\OWKd0i4\\OWKd0i4iNWKQN0V3i4hNXKRNOV3i4hNYKQNNX3h4gN[KPNMY3h4gN[KPNMY3h4fN]KPNKZ3h4fN^KoMJ\\3g4eN`KnMI]3h4dN_KoMI]3h4cNaKoMG^3h4cNbKnMF`3g4bNcKnMF`3g4bNdKmMEa3g4aNfKmMCb3g4aNgKlMBd3f4`NhKlMBd3f4`NiKkMAe3f4_NkKkM_Of3f4_NkKkM_Og3e4^NmKjM^Oh3e4^NnKiM]Oi3e4]NPLiM[Oj3e4]NPLiM[Ok3d4\\NRLhMZOl3d4\\NSLgMYOm3d4[NVLf1h3[NXLe1h3[NXLf1g3ZNYLf1f3ZN[Lg1d3YN\\Lh1c3XN]Lh1b3YN^Lh1a3WN`Li1`3WN`Lj1_3VNaLk1]3VNcLj1]3UNdLl1[3TNeLl1Z3UNfLl1Y3TNgLm1X3RNiLo1U3RNkLn1U3RNkLo1S3QNnLP2Q3PNoLP2Q3PNoLQ2o2PNQMQ2n2nMSMS2k2nMUMS2j2mMVMT2h2mMXMT2g2kMZMV2c2lM]MU2a2lM_MU2_2kMbMV2[2lMeMU2Y2lMgMU2W2kMjMV2T2kMlMW2P2kMPNV2n1jMRNY2j1iMVNX2g1jMYNX2c1iM^NY2^1iMbNX2[1jMeNX2T1kJiMn2T1[2m0iJnMm2U1]2h0hJSNk2U1`2c0gJXNh2V1d2<gJ^Ne2V1h26eJdNc2V1_MjNn4V1RKjN`2W1^MQOn4f0XKRO\\2W1]MYOl45`K[OW2W1[MAm4DeKDR2X1[MHo5POfLX1XM1l0hNm3OoMX1TM`0?_N^4InMk3R2ULnMk3R2ULnMk3R2ULmMl3S2TLlMm38SLc1n3lLfMP3[2TMdMh2]2\\MbM`2_2dMaMW2`2nM^Mn1c2VN\\Mf1e2^NZM_1f2fNYMU1h2POSMo0n2[OhLd0\\3i33M3K5L5J5L4XOh0N3M2N2N2N2N2N2N2N2N1O3M2N3M3L3N3L4L3N5J7I8Ha2`M^Wk0",
                },
                {
                    "size": [375, 500],
                    "counts": b"nn]12e;1O2L4M2M3N2N2N2N2N2N3M2O1N3N1N2O1O1N101N2O1O0O2O1N2O1N2O1N1[F^NT9d1kFeNk8]1TGeNi8X2L3M3L4M2N1O2N1O1O2N1N3M2O1N2N2N2O1N2N2O001O1O1O1O1O001O1O0001O001O001O001M2N3M2M4H7J7O0101N101N2O2M3N1N2O1N010O011O001VH[L`7e3`H]L^7d3aH]L^7c3bH^L]7c3bH^L]7b3cH_L\\7b3cH_L\\7b3cH^L\\7c3dH]L\\7d3cH\\L]7e3bH\\L]7d3dH[L\\7f3cHZL]7g3bHYL^7g3bHZL]7P4000001O0O101O00001O0000001O0000001O0000001O000000001O0000001O000000ZOTIaLl6_3TIbLl6\\3UIdLk6\\3UIdLk6[3VIeLj6[3VIfLi6Y3XIgLh6Y3XIgLh6X3YIhLg6X3YIhLg6W3ZIiLf6W3ZIiLg6U3ZIlLe6T3[IlLe6U3ZIkLf6U3ZIkLf6V3YIjLg6V3YIjLg6W3XIiLh6X3WIiLh6W3XIiLh6X3WIhLj6X3UIhLk6X3UIhLk6Y3TIgLl6Z3TIeLl6\\3SIeLl6[3TIeLl6[3TIeLm6[3RIeLn6[3RIeLn6[3RIeLo6Z3QIfLo6Z3QIgLn6Z3QIfLP7Y3PIgLP7Y3PIgLP7Z3oHfLR7Y3nHgLR7Z3mHfLS7[3lHeLT7[3lHeLU7[3jHeLV7[3jHeLV7n31O00001O00001O00001O00001O000O2O00001N101O0O101O0O101N10001N2O1N2O1N2O1N2O1N2N2N3M2N2N2M4M2N3M3M3K5K5J6K5J6K4N3M3L4M7I7E:C>BWZ^1",
                },
            ]
        ),
    }
    assert record == expected
