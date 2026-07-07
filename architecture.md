# Sơ đồ Kiến trúc & Luồng Forward của YOLOv10-Lite

Tài liệu này mô tả chi tiết luồng xử lý đồng thời (`forward`) của mô hình YOLOv10-Lite, từ ảnh đầu vào gốc kích thước $480 \times 480$ pixel cho tới đầu ra dự đoán.

---

## 1. Sơ đồ Mermaid Chi Tiết

```mermaid
graph TD
    %% Định nghĩa phong cách các khối
    classDef input_output fill:#f8cecc,stroke:#b85450,stroke-width:2px;
    classDef conv_layer fill:#dae8fc,stroke:#6c8ebf,stroke-width:1px;
    classDef bn_layer fill:#fff2cc,stroke:#d6b656,stroke-width:1px;
    classDef act_layer fill:#d5e8d4,stroke:#82b366,stroke-width:1px;
    classDef concat_split fill:#e1d5e7,stroke:#9673a6,stroke-width:1px;
    classDef pool_layer fill:#ffe6cc,stroke:#d79b00,stroke-width:1px;

    %% --- ĐẦU VÀO ---
    Input(["Input Image: [B, 3, 480, 480]"]):::input_output

    %% =========================================================================
    %% 1. BACKBONE (CSPDarknet-Lite)
    %% =========================================================================
    subgraph Backbone ["1. Backbone (CSPDarknet-Lite)"]
        %% Stem Conv
        subgraph Stem ["Stem Layer"]
            Stem_Conv["Conv2D: 3 -> c0, k3, s2"]:::conv_layer
            Stem_BN["BatchNorm2d: c0"]:::bn_layer
            Stem_Act["SiLU"]:::act_layer
            Stem_Conv --> Stem_BN --> Stem_Act
        end

        %% Stage 1
        subgraph Stage1 ["Stage 1"]
            S1_Down_Conv["Conv2D: c0 -> c1, k3, s2"]:::conv_layer
            S1_Down_BN["BatchNorm2d: c1"]:::bn_layer
            S1_Down_Act["SiLU"]:::act_layer
            
            subgraph S1_C2f ["C2f (n=1, shortcut=True)"]
                S1_C2f_cv1_Conv["Conv2D: c1 -> 2*c, k1, s1"]:::conv_layer
                S1_C2f_cv1_BN["BatchNorm2d: 2*c"]:::bn_layer
                S1_C2f_cv1_Act["SiLU"]:::act_layer
                S1_C2f_Split["Split: Chunk (2)"]:::concat_split
                
                subgraph S1_C2f_B1 ["Bottleneck 1"]
                    S1_C2f_B1_cv1_Conv["Conv2D: c -> c, k3, s1"]:::conv_layer
                    S1_C2f_B1_cv1_BN["BatchNorm2d: c"]:::bn_layer
                    S1_C2f_B1_cv1_Act["SiLU"]:::act_layer
                    S1_C2f_B1_cv2_Conv["Conv2D: c -> c, k3, s1"]:::conv_layer
                    S1_C2f_B1_cv2_BN["BatchNorm2d: c"]:::bn_layer
                    S1_C2f_B1_cv2_Act["SiLU"]:::act_layer
                    S1_C2f_B1_Add["Element-wise Add (+)"]:::concat_split
                    
                    S1_C2f_B1_cv1_Conv --> S1_C2f_B1_cv1_BN --> S1_C2f_B1_cv1_Act
                    S1_C2f_B1_cv1_Act --> S1_C2f_B1_cv2_Conv --> S1_C2f_B1_cv2_BN --> S1_C2f_B1_cv2_Act
                end
                
                S1_C2f_Concat["Concat: [y0, y1, y2]"]:::concat_split
                S1_C2f_cv2_Conv["Conv2D: 3*c -> c1, k1, s1"]:::conv_layer
                S1_C2f_cv2_BN["BatchNorm2d: c1"]:::bn_layer
                S1_C2f_cv2_Act["SiLU"]:::act_layer
            end
            
            S1_Down_Conv --> S1_Down_BN --> S1_Down_Act --> S1_C2f_cv1_Conv
            S1_C2f_cv1_Conv --> S1_C2f_cv1_BN --> S1_C2f_cv1_Act --> S1_C2f_Split
            S1_C2f_Split -->|y0| S1_C2f_Concat
            S1_C2f_Split -->|y1| S1_C2f_B1_cv1_Conv
            S1_C2f_Split -->|y1| S1_C2f_B1_Add
            S1_C2f_B1_cv2_Act --> S1_C2f_B1_Add
            S1_C2f_B1_Add -->|y2| S1_C2f_Concat
            S1_C2f_Concat --> S1_C2f_cv2_Conv --> S1_C2f_cv2_BN --> S1_C2f_cv2_Act
        end

        %% Stage 2 (P3)
        subgraph Stage2 ["Stage 2 (P3)"]
            S2_Down_Conv["Conv2D: c1 -> c2, k3, s2"]:::conv_layer
            S2_Down_BN["BatchNorm2d: c2"]:::bn_layer
            S2_Down_Act["SiLU"]:::act_layer
            
            subgraph S2_C2f ["C2f (n=2, shortcut=True)"]
                S2_C2f_cv1_Conv["Conv2D: c2 -> 2*c, k1, s1"]:::conv_layer
                S2_C2f_cv1_BN["BatchNorm2d: 2*c"]:::bn_layer
                S2_C2f_cv1_Act["SiLU"]:::act_layer
                S2_C2f_Split["Split: Chunk (2)"]:::concat_split
                
                subgraph S2_C2f_B1 ["Bottleneck 1"]
                    S2_C2f_B1_cv1_Conv["Conv2D: c -> c, k3, s1"]:::conv_layer
                    S2_C2f_B1_cv1_BN["BatchNorm2d: c"]:::bn_layer
                    S2_C2f_B1_cv1_Act["SiLU"]:::act_layer
                    S2_C2f_B1_cv2_Conv["Conv2D: c -> c, k3, s1"]:::conv_layer
                    S2_C2f_B1_cv2_BN["BatchNorm2d: c"]:::bn_layer
                    S2_C2f_B1_cv2_Act["SiLU"]:::act_layer
                    S2_C2f_B1_Add["Element-wise Add (+)"]:::concat_split
                end
                
                subgraph S2_C2f_B2 ["Bottleneck 2"]
                    S2_C2f_B2_cv1_Conv["Conv2D: c -> c, k3, s1"]:::conv_layer
                    S2_C2f_B2_cv1_BN["BatchNorm2d: c"]:::bn_layer
                    S2_C2f_B2_cv1_Act["SiLU"]:::act_layer
                    S2_C2f_B2_cv2_Conv["Conv2D: c -> c, k3, s1"]:::conv_layer
                    S2_C2f_B2_cv2_BN["BatchNorm2d: c"]:::bn_layer
                    S2_C2f_B2_cv2_Act["SiLU"]:::act_layer
                    S2_C2f_B2_Add["Element-wise Add (+)"]:::concat_split
                end
                
                S2_C2f_Concat["Concat: [y0, y1, y2, y3]"]:::concat_split
                S2_C2f_cv2_Conv["Conv2D: 4*c -> c2, k1, s1"]:::conv_layer
                S2_C2f_cv2_BN["BatchNorm2d: c2"]:::bn_layer
                S2_C2f_cv2_Act["SiLU"]:::act_layer
            end
            
            S2_Down_Conv --> S2_Down_BN --> S2_Down_Act --> S2_C2f_cv1_Conv
            S2_C2f_cv1_Conv --> S2_C2f_cv1_BN --> S2_C2f_cv1_Act --> S2_C2f_Split
            S2_C2f_Split -->|y0| S2_C2f_Concat
            S2_C2f_Split -->|y1| S2_C2f_B1_cv1_Conv
            S2_C2f_Split -->|y1| S2_C2f_B1_Add
            S2_C2f_B1_cv1_Conv --> S2_C2f_B1_cv1_BN --> S2_C2f_B1_cv1_Act --> S2_C2f_B1_cv2_Conv --> S2_C2f_B1_cv2_BN --> S2_C2f_B1_cv2_Act --> S2_C2f_B1_Add
            S2_C2f_B1_Add -->|y2| S2_C2f_B2_cv1_Conv
            S2_C2f_B1_Add -->|y2| S2_C2f_B2_Add
            S2_C2f_B1_Add -->|y2| S2_C2f_Concat
            S2_C2f_B2_cv1_Conv --> S2_C2f_B2_cv1_BN --> S2_C2f_B2_cv1_Act --> S2_C2f_B2_cv2_Conv --> S2_C2f_B2_cv2_BN --> S2_C2f_B2_cv2_Act --> S2_C2f_B2_Add
            S2_C2f_B2_Add -->|y3| S2_C2f_Concat
            S2_C2f_Concat --> S2_C2f_cv2_Conv --> S2_C2f_cv2_BN --> S2_C2f_cv2_Act
        end

        %% Stage 3 (P4)
        subgraph Stage3 ["Stage 3 (P4)"]
            S3_Down_Conv["Conv2D: c2 -> c3, k3, s2"]:::conv_layer
            S3_Down_BN["BatchNorm2d: c3"]:::bn_layer
            S3_Down_Act["SiLU"]:::act_layer
            
            subgraph S3_C2f ["C2f (n=2, shortcut=True)"]
                S3_C2f_cv1_Conv["Conv2D: c3 -> 2*c, k1, s1"]:::conv_layer
                S3_C2f_cv1_BN["BatchNorm2d: 2*c"]:::bn_layer
                S3_C2f_cv1_Act["SiLU"]:::act_layer
                S3_C2f_Split["Split: Chunk (2)"]:::concat_split
                
                subgraph S3_C2f_B1 ["Bottleneck 1"]
                    S3_C2f_B1_cv1_Conv["Conv2D: c -> c, k3, s1"]:::conv_layer
                    S3_C2f_B1_cv1_BN["BatchNorm2d: c"]:::bn_layer
                    S3_C2f_B1_cv1_Act["SiLU"]:::act_layer
                    S3_C2f_B1_cv2_Conv["Conv2D: c -> c, k3, s1"]:::conv_layer
                    S3_C2f_B1_cv2_BN["BatchNorm2d: c"]:::bn_layer
                    S3_C2f_B1_cv2_Act["SiLU"]:::act_layer
                    S3_C2f_B1_Add["Element-wise Add (+)"]:::concat_split
                end
                
                subgraph S3_C2f_B2 ["Bottleneck 2"]
                    S3_C2f_B2_cv1_Conv["Conv2D: c -> c, k3, s1"]:::conv_layer
                    S3_C2f_B2_cv1_BN["BatchNorm2d: c"]:::bn_layer
                    S3_C2f_B2_cv1_Act["SiLU"]:::act_layer
                    S3_C2f_B2_cv2_Conv["Conv2D: c -> c, k3, s1"]:::conv_layer
                    S3_C2f_B2_cv2_BN["BatchNorm2d: c"]:::bn_layer
                    S3_C2f_B2_cv2_Act["SiLU"]:::act_layer
                    S3_C2f_B2_Add["Element-wise Add (+)"]:::concat_split
                end
                
                S3_C2f_Concat["Concat: [y0, y1, y2, y3]"]:::concat_split
                S3_C2f_cv2_Conv["Conv2D: 4*c -> c3, k1, s1"]:::conv_layer
                S3_C2f_cv2_BN["BatchNorm2d: c3"]:::bn_layer
                S3_C2f_cv2_Act["SiLU"]:::act_layer
            end
            
            S3_Down_Conv --> S3_Down_BN --> S3_Down_Act --> S3_C2f_cv1_Conv
            S3_C2f_cv1_Conv --> S3_C2f_cv1_BN --> S3_C2f_cv1_Act --> S3_C2f_Split
            S3_C2f_Split -->|y0| S3_C2f_Concat
            S3_C2f_Split -->|y1| S3_C2f_B1_cv1_Conv
            S3_C2f_Split -->|y1| S3_C2f_B1_Add
            S3_C2f_B1_cv1_Conv --> S3_C2f_B1_cv1_BN --> S3_C2f_B1_cv1_Act --> S3_C2f_B1_cv2_Conv --> S3_C2f_B1_cv2_BN --> S3_C2f_B1_cv2_Act --> S3_C2f_B1_Add
            S3_C2f_B1_Add -->|y2| S3_C2f_B2_cv1_Conv
            S3_C2f_B1_Add -->|y2| S3_C2f_B2_Add
            S3_C2f_B1_Add -->|y2| S3_C2f_Concat
            S3_C2f_B2_cv1_Conv --> S3_C2f_B2_cv1_BN --> S3_C2f_B2_cv1_Act --> S3_C2f_B2_cv2_Conv --> S3_C2f_B2_cv2_BN --> S3_C2f_B2_cv2_Act --> S3_C2f_B2_Add
            S3_C2f_B2_Add -->|y3| S3_C2f_Concat
            S3_C2f_Concat --> S3_C2f_cv2_Conv --> S3_C2f_cv2_BN --> S3_C2f_cv2_Act
        end

        %% Stage 4 (P5)
        subgraph Stage4 ["Stage 4 (P5)"]
            S4_Down_Conv["Conv2D: c3 -> c4, k3, s2"]:::conv_layer
            S4_Down_BN["BatchNorm2d: c4"]:::bn_layer
            S4_Down_Act["SiLU"]:::act_layer
            
            subgraph S4_C2f ["C2f (n=1, shortcut=True)"]
                S4_C2f_cv1_Conv["Conv2D: c4 -> 2*c, k1, s1"]:::conv_layer
                S4_C2f_cv1_BN["BatchNorm2d: 2*c"]:::bn_layer
                S4_C2f_cv1_Act["SiLU"]:::act_layer
                S4_C2f_Split["Split: Chunk (2)"]:::concat_split
                
                subgraph S4_C2f_B1 ["Bottleneck 1"]
                    S4_C2f_B1_cv1_Conv["Conv2D: c -> c, k3, s1"]:::conv_layer
                    S4_C2f_B1_cv1_BN["BatchNorm2d: c"]:::bn_layer
                    S4_C2f_B1_cv1_Act["SiLU"]:::act_layer
                    S4_C2f_B1_cv2_Conv["Conv2D: c -> c, k3, s1"]:::conv_layer
                    S4_C2f_B1_cv2_BN["BatchNorm2d: c"]:::bn_layer
                    S4_C2f_B1_cv2_Act["SiLU"]:::act_layer
                    S4_C2f_B1_Add["Element-wise Add (+)"]:::concat_split
                end
                
                S4_C2f_Concat["Concat: [y0, y1, y2]"]:::concat_split
                S4_C2f_cv2_Conv["Conv2D: 3*c -> c4, k1, s1"]:::conv_layer
                S4_C2f_cv2_BN["BatchNorm2d: c4"]:::bn_layer
                S4_C2f_cv2_Act["SiLU"]:::act_layer
            end
            
            subgraph S4_SPPF ["SPPF (k=5)"]
                SPPF_cv1_Conv["Conv2D: c4 -> c_, k1, s1"]:::conv_layer
                SPPF_cv1_BN["BatchNorm2d: c_"]:::bn_layer
                SPPF_cv1_Act["SiLU"]:::act_layer
                SPPF_MP1["MaxPool2D: k5, s1, p2"]:::pool_layer
                SPPF_MP2["MaxPool2D: k5, s1, p2"]:::pool_layer
                SPPF_MP3["MaxPool2D: k5, s1, p2"]:::pool_layer
                SPPF_Concat["Concat: [x, MP1, MP2, MP3]"]:::concat_split
                SPPF_cv2_Conv["Conv2D: 4*c_ -> c4, k1, s1"]:::conv_layer
                SPPF_cv2_BN["BatchNorm2d: c4"]:::bn_layer
                SPPF_cv2_Act["SiLU"]:::act_layer
                
                SPPF_cv1_Conv --> SPPF_cv1_BN --> SPPF_cv1_Act
                SPPF_cv1_Act --> SPPF_MP1 --> SPPF_MP2 --> SPPF_MP3
                SPPF_cv1_Act --> SPPF_Concat
                SPPF_MP1 --> SPPF_Concat
                SPPF_MP2 --> SPPF_Concat
                SPPF_MP3 --> SPPF_Concat
                SPPF_Concat --> SPPF_cv2_Conv --> SPPF_cv2_BN --> SPPF_cv2_Act
            end
            
            S4_Down_Conv --> S4_Down_BN --> S4_Down_Act --> S4_C2f_cv1_Conv
            S4_C2f_cv1_Conv --> S4_C2f_cv1_BN --> S4_C2f_cv1_Act --> S4_C2f_Split
            S4_C2f_Split -->|y0| S4_C2f_Concat
            S4_C2f_Split -->|y1| S4_C2f_B1_cv1_Conv
            S4_C2f_Split -->|y1| S4_C2f_B1_Add
            S4_C2f_B1_cv1_Conv --> S4_C2f_B1_cv1_BN --> S4_C2f_B1_cv1_Act --> S4_C2f_B1_cv2_Conv --> S4_C2f_B1_cv2_BN --> S4_C2f_B1_cv2_Act --> S4_C2f_B1_Add
            S4_C2f_B1_Add -->|y2| S4_C2f_Concat
            S4_C2f_Concat --> S4_C2f_cv2_Conv --> S4_C2f_cv2_BN --> S4_C2f_cv2_Act --> SPPF_cv1_Conv
        end
    end

    Stem_Act --> S1_Down_Conv
    S1_C2f_cv2_Act --> S2_Down_Conv
    S2_C2f_cv2_Act --> S3_Down_Conv
    S3_C2f_cv2_Act --> S4_Down_Conv

    %% =========================================================================
    %% 2. NECK (PAFPN)
    %% =========================================================================
    subgraph Neck ["2. Neck (PAFPN Feature Fusion)"]
        %% --- Đường đi Top-Down (FPN) ---
        subgraph FPN ["Top-Down Pathway"]
            N_Reduce5_Conv["Conv2D: c5 -> c4, k1, s1"]:::conv_layer
            N_Reduce5_BN["BatchNorm2d: c4"]:::bn_layer
            N_Reduce5_Act["SiLU"]:::act_layer
            N_Up5["Upsample: scale_factor=2"]:::pool_layer
            N_Concat4["Concat: [P5_Up, P4]"]:::concat_split
            
            subgraph N_c2f_p4 ["c2f_p4 (n=1, shortcut=False)"]
                N_p4_cv1_Conv["Conv2D: 2*c4 -> 2*c, k1, s1"]:::conv_layer
                N_p4_cv1_BN["BatchNorm2d: 2*c"]:::bn_layer
                N_p4_cv1_Act["SiLU"]:::act_layer
                N_p4_Split["Split: Chunk (2)"]:::concat_split
                subgraph N_p4_B1 ["Bottleneck (No shortcut)"]
                    N_p4_B1_cv1_Conv["Conv2D: c -> c, k3, s1"]:::conv_layer
                    N_p4_B1_cv1_BN["BatchNorm2d: c"]:::bn_layer
                    N_p4_B1_cv1_Act["SiLU"]:::act_layer
                    N_p4_B1_cv2_Conv["Conv2D: c -> c, k3, s1"]:::conv_layer
                    N_p4_B1_cv2_BN["BatchNorm2d: c"]:::bn_layer
                    N_p4_B1_cv2_Act["SiLU"]:::act_layer
                end
                N_p4_Concat["Concat: [y0, y1, y2]"]:::concat_split
                N_p4_cv2_Conv["Conv2D: 3*c -> c4, k1, s1"]:::conv_layer
                N_p4_cv2_BN["BatchNorm2d: c4"]:::bn_layer
                N_p4_cv2_Act["SiLU"]:::act_layer
            end
            
            N_Reduce4_Conv["Conv2D: c4 -> c3, k1, s1"]:::conv_layer
            N_Reduce4_BN["BatchNorm2d: c3"]:::bn_layer
            N_Reduce4_Act["SiLU"]:::act_layer
            N_Up4["Upsample: scale_factor=2"]:::pool_layer
            N_Concat3["Concat: [P4_Up, P3]"]:::concat_split
            
            subgraph N_c2f_p3 ["c2f_p3 (n=1, shortcut=False)"]
                N_p3_cv1_Conv["Conv2D: 2*c3 -> 2*c, k1, s1"]:::conv_layer
                N_p3_cv1_BN["BatchNorm2d: 2*c"]:::bn_layer
                N_p3_cv1_Act["SiLU"]:::act_layer
                N_p3_Split["Split: Chunk (2)"]:::concat_split
                subgraph N_p3_B1 ["Bottleneck (No shortcut)"]
                    N_p3_B1_cv1_Conv["Conv2D: c -> c, k3, s1"]:::conv_layer
                    N_p3_B1_cv1_BN["BatchNorm2d: c"]:::bn_layer
                    N_p3_B1_cv1_Act["SiLU"]:::act_layer
                    N_p3_B1_cv2_Conv["Conv2D: c -> c, k3, s1"]:::conv_layer
                    N_p3_B1_cv2_BN["BatchNorm2d: c"]:::bn_layer
                    N_p3_B1_cv2_Act["SiLU"]:::act_layer
                end
                N_p3_Concat["Concat: [y0, y1, y2]"]:::concat_split
                N_p3_cv2_Conv["Conv2D: 3*c -> c3, k1, s1"]:::conv_layer
                N_p3_cv2_BN["BatchNorm2d: c3"]:::bn_layer
                N_p3_cv2_Act["SiLU"]:::act_layer
            end
        end

        %% --- Đường đi Bottom-Up (PAN) ---
        subgraph PAN ["Bottom-Up Pathway"]
            N_Down3_Conv["Conv2D: c3 -> c3, k3, s2"]:::conv_layer
            N_Down3_BN["BatchNorm2d: c3"]:::bn_layer
            N_Down3_Act["SiLU"]:::act_layer
            N_Concat_n4["Concat: [P3_Down, P4_Red]"]:::concat_split
            
            subgraph N_c2f_n4 ["c2f_n4 (n=1, shortcut=False)"]
                N_n4_cv1_Conv["Conv2D: 2*c3 -> 2*c, k1, s1"]:::conv_layer
                N_n4_cv1_BN["BatchNorm2d: 2*c"]:::bn_layer
                N_n4_cv1_Act["SiLU"]:::act_layer
                N_n4_Split["Split: Chunk (2)"]:::concat_split
                subgraph N_n4_B1 ["Bottleneck (No shortcut)"]
                    N_n4_B1_cv1_Conv["Conv2D: c -> c, k3, s1"]:::conv_layer
                    N_n4_B1_cv1_BN["BatchNorm2d: c"]:::bn_layer
                    N_n4_B1_cv1_Act["SiLU"]:::act_layer
                    N_n4_B1_cv2_Conv["Conv2D: c -> c, k3, s1"]:::conv_layer
                    N_n4_B1_cv2_BN["BatchNorm2d: c"]:::bn_layer
                    N_n4_B1_cv2_Act["SiLU"]:::act_layer
                end
                N_n4_Concat["Concat: [y0, y1, y2]"]:::concat_split
                N_n4_cv2_Conv["Conv2D: 3*c -> c4, k1, s1"]:::conv_layer
                N_n4_cv2_BN["BatchNorm2d: c4"]:::bn_layer
                N_n4_cv2_Act["SiLU"]:::act_layer
            end

            N_Down4_Conv["Conv2D: c4 -> c4, k3, s2"]:::conv_layer
            N_Down4_BN["BatchNorm2d: c4"]:::bn_layer
            N_Down4_Act["SiLU"]:::act_layer
            N_Concat_n5["Concat: [P4_Down, P5_Red]"]:::concat_split

            subgraph N_c2f_n5 ["c2f_n5 (n=1, shortcut=False)"]
                N_n5_cv1_Conv["Conv2D: 2*c4 -> 2*c, k1, s1"]:::conv_layer
                N_n5_cv1_BN["BatchNorm2d: 2*c"]:::bn_layer
                N_n5_cv1_Act["SiLU"]:::act_layer
                N_n5_Split["Split: Chunk (2)"]:::concat_split
                subgraph N_n5_B1 ["Bottleneck (No shortcut)"]
                    N_n5_B1_cv1_Conv["Conv2D: c -> c, k3, s1"]:::conv_layer
                    N_n5_B1_cv1_BN["BatchNorm2d: c"]:::bn_layer
                    N_n5_B1_cv1_Act["SiLU"]:::act_layer
                    N_n5_B1_cv2_Conv["Conv2D: c -> c, k3, s1"]:::conv_layer
                    N_n5_B1_cv2_BN["BatchNorm2d: c"]:::bn_layer
                    N_n5_B1_cv2_Act["SiLU"]:::act_layer
                end
                N_n5_Concat["Concat: [y0, y1, y2]"]:::concat_split
                N_n5_cv2_Conv["Conv2D: 3*c -> c5, k1, s1"]:::conv_layer
                N_n5_cv2_BN["BatchNorm2d: c5"]:::bn_layer
                N_n5_cv2_Act["SiLU"]:::act_layer
            end
        end
    end

    %% Kết nối từ Backbone sang Neck
    SPPF_cv2_Act --> N_Reduce5_Conv
    S3_C2f_cv2_Act --> N_Concat4
    S2_C2f_cv2_Act --> N_Concat3

    %% Flow Top-Down
    N_Reduce5_Conv --> N_Reduce5_BN --> N_Reduce5_Act --> N_Up5 --> N_Concat4
    N_Concat4 --> N_p4_cv1_Conv
    N_p4_cv1_Conv --> N_p4_cv1_BN --> N_p4_cv1_Act --> N_p4_Split
    N_p4_Split -->|y0| N_p4_Concat
    N_p4_Split -->|y1| N_p4_B1_cv1_Conv
    N_p4_B1_cv1_Conv --> N_p4_B1_cv1_BN --> N_p4_B1_cv1_Act --> N_p4_B1_cv2_Conv --> N_p4_B1_cv2_BN --> N_p4_B1_cv2_Act --> N_p4_Concat
    N_p4_Concat --> N_p4_cv2_Conv --> N_p4_cv2_BN --> N_p4_cv2_Act --> N_Reduce4_Conv
    
    N_Reduce4_Conv --> N_Reduce4_BN --> N_Reduce4_Act --> N_Up4 --> N_Concat3
    N_Concat3 --> N_p3_cv1_Conv
    N_p3_cv1_Conv --> N_p3_cv1_BN --> N_p3_cv1_Act --> N_p3_Split
    N_p3_Split -->|y0| N_p3_Concat
    N_p3_Split -->|y1| N_p3_B1_cv1_Conv
    N_p3_B1_cv1_Conv --> N_p3_B1_cv1_BN --> N_p3_B1_cv1_Act --> N_p3_B1_cv2_Conv --> N_p3_B1_cv2_BN --> N_p3_B1_cv2_Act --> N_p3_Concat
    N_p3_Concat --> N_p3_cv2_Conv --> N_p3_cv2_BN --> N_p3_cv2_Act

    %% Flow Bottom-Up
    N_p3_cv2_Act --> N_Down3_Conv
    N_Down3_Conv --> N_Down3_BN --> N_Down3_Act --> N_Concat_n4
    N_Reduce4_Act --> N_Concat_n4
    
    N_Concat_n4 --> N_n4_cv1_Conv
    N_n4_cv1_Conv --> N_n4_cv1_BN --> N_n4_cv1_Act --> N_n4_Split
    N_n4_Split -->|y0| N_n4_Concat
    N_n4_Split -->|y1| N_n4_B1_cv1_Conv
    N_n4_B1_cv1_Conv --> N_n4_B1_cv1_BN --> N_n4_B1_cv1_Act --> N_n4_B1_cv2_Conv --> N_n4_B1_cv2_BN --> N_n4_B1_cv2_Act --> N_n4_Concat
    N_n4_Concat --> N_n4_cv2_Conv --> N_n4_cv2_BN --> N_n4_cv2_Act --> N_Down4_Conv
    
    N_Down4_Conv --> N_Down4_BN --> N_Down4_Act --> N_Concat_n5
    N_Reduce5_Act --> N_Concat_n5
    
    N_Concat_n5 --> N_n5_cv1_Conv
    N_n5_cv1_Conv --> N_n5_cv1_BN --> N_n5_cv1_Act --> N_n5_Split
    N_n5_Split -->|y0| N_n5_Concat
    N_n5_Split -->|y1| N_n5_B1_cv1_Conv
    N_n5_B1_cv1_Conv --> N_n5_B1_cv1_BN --> N_n5_B1_cv1_Act --> N_n5_B1_cv2_Conv --> N_n5_B1_cv2_BN --> N_n5_B1_cv2_Act --> N_n5_Concat
    N_n5_Concat --> N_n5_cv2_Conv --> N_n5_cv2_BN --> N_n5_cv2_Act

    %% =========================================================================
    %% 3. DETECT HEAD
    %% =========================================================================
    subgraph Head ["3. Detect Head (ScaleHeads & Dual Branches)"]
        %% Level 1: P3_out (60x60)
        subgraph L1_Head ["ScaleHead 1 (P3_out, channels=c3)"]
            %% Cls Stem (DW Conv & Standard Conv)
            L1_cls_DW1_Conv["DWConv2D: c3 -> c3, k3, s1"]:::conv_layer
            L1_cls_DW1_BN["BatchNorm2d: c3"]:::bn_layer
            L1_cls_DW1_Act["SiLU"]:::act_layer
            L1_cls_C1_Conv["Conv2D: c3 -> c_cls, k1, s1"]:::conv_layer
            L1_cls_C1_BN["BatchNorm2d: c_cls"]:::bn_layer
            L1_cls_C1_Act["SiLU"]:::act_layer
            L1_cls_DW2_Conv["DWConv2D: c_cls -> c_cls, k3, s1"]:::conv_layer
            L1_cls_DW2_BN["BatchNorm2d: c_cls"]:::bn_layer
            L1_cls_DW2_Act["SiLU"]:::act_layer
            L1_cls_C2_Conv["Conv2D: c_cls -> c_cls, k1, s1"]:::conv_layer
            L1_cls_C2_BN["BatchNorm2d: c_cls"]:::bn_layer
            L1_cls_C2_Act["SiLU"]:::act_layer
            
            L1_cls_DW1_Conv --> L1_cls_DW1_BN --> L1_cls_DW1_Act --> L1_cls_C1_Conv --> L1_cls_C1_BN --> L1_cls_C1_Act
            L1_cls_C1_Act --> L1_cls_DW2_Conv --> L1_cls_DW2_BN --> L1_cls_DW2_Act --> L1_cls_C2_Conv --> L1_cls_C2_BN --> L1_cls_C2_Act

            %% Reg Stem (Standard Conv)
            L1_reg_C1_Conv["Conv2D: c3 -> c_reg, k3, s1"]:::conv_layer
            L1_reg_C1_BN["BatchNorm2d: c_reg"]:::bn_layer
            L1_reg_C1_Act["SiLU"]:::act_layer
            L1_reg_C2_Conv["Conv2D: c_reg -> c_reg, k3, s1"]:::conv_layer
            L1_reg_C2_BN["BatchNorm2d: c_reg"]:::bn_layer
            L1_reg_C2_Act["SiLU"]:::act_layer
            
            L1_reg_C1_Conv --> L1_reg_C1_BN --> L1_reg_C1_Act --> L1_reg_C2_Conv --> L1_reg_C2_BN --> L1_reg_C2_Act

            %% Predictors 1x1 Conv
            L1_cls_o2m["Conv2D: c_cls -> nc, k1"]:::conv_layer
            L1_reg_o2m["Conv2D: c_reg -> 4*reg_max, k1"]:::conv_layer
            L1_cls_o2o["Conv2D: c_cls -> nc, k1"]:::conv_layer
            L1_reg_o2o["Conv2D: c_reg -> 4*reg_max, k1"]:::conv_layer
            
            L1_cls_C2_Act --> L1_cls_o2m
            L1_cls_C2_Act --> L1_cls_o2o
            L1_reg_C2_Act --> L1_reg_o2m
            L1_reg_C2_Act --> L1_reg_o2o
        end

        %% Level 2: P4_out (30x30)
        subgraph L2_Head ["ScaleHead 2 (P4_out, channels=c4)"]
            L2_cls_DW1_Conv["DWConv2D: c4 -> c4, k3, s1"]:::conv_layer
            L2_cls_DW1_BN["BatchNorm2d: c4"]:::bn_layer
            L2_cls_DW1_Act["SiLU"]:::act_layer
            L2_cls_C1_Conv["Conv2D: c4 -> c_cls, k1, s1"]:::conv_layer
            L2_cls_C1_BN["BatchNorm2d: c_cls"]:::bn_layer
            L2_cls_C1_Act["SiLU"]:::act_layer
            L2_cls_DW2_Conv["DWConv2D: c_cls -> c_cls, k3, s1"]:::conv_layer
            L2_cls_DW2_BN["BatchNorm2d: c_cls"]:::bn_layer
            L2_cls_DW2_Act["SiLU"]:::act_layer
            L2_cls_C2_Conv["Conv2D: c_cls -> c_cls, k1, s1"]:::conv_layer
            L2_cls_C2_BN["BatchNorm2d: c_cls"]:::bn_layer
            L2_cls_C2_Act["SiLU"]:::act_layer
            
            L2_cls_DW1_Conv --> L2_cls_DW1_BN --> L2_cls_DW1_Act --> L2_cls_C1_Conv --> L2_cls_C1_BN --> L2_cls_C1_Act
            L2_cls_C1_Act --> L2_cls_DW2_Conv --> L2_cls_DW2_BN --> L2_cls_DW2_Act --> L2_cls_C2_Conv --> L2_cls_C2_BN --> L2_cls_C2_Act

            L2_reg_C1_Conv["Conv2D: c4 -> c_reg, k3, s1"]:::conv_layer
            L2_reg_C1_BN["BatchNorm2d: c_reg"]:::bn_layer
            L2_reg_C1_Act["SiLU"]:::act_layer
            L2_reg_C2_Conv["Conv2D: c_reg -> c_reg, k3, s1"]:::conv_layer
            L2_reg_C2_BN["BatchNorm2d: c_reg"]:::bn_layer
            L2_reg_C2_Act["SiLU"]:::act_layer
            
            L2_reg_C1_Conv --> L2_reg_C1_BN --> L2_reg_C1_Act --> L2_reg_C2_Conv --> L2_reg_C2_BN --> L2_reg_C2_Act

            L2_cls_o2m["Conv2D: c_cls -> nc, k1"]:::conv_layer
            L2_reg_o2m["Conv2D: c_reg -> 4*reg_max, k1"]:::conv_layer
            L2_cls_o2o["Conv2D: c_cls -> nc, k1"]:::conv_layer
            L2_reg_o2o["Conv2D: c_reg -> 4*reg_max, k1"]:::conv_layer
            
            L2_cls_C2_Act --> L2_cls_o2m
            L2_cls_C2_Act --> L2_cls_o2o
            L2_reg_C2_Act --> L2_reg_o2m
            L2_reg_C2_Act --> L2_reg_o2o
        end

        %% Level 3: P5_out (15x15)
        subgraph L3_Head ["ScaleHead 3 (P5_out, channels=c5)"]
            L3_cls_DW1_Conv["DWConv2D: c5 -> c5, k3, s1"]:::conv_layer
            L3_cls_DW1_BN["BatchNorm2d: c5"]:::bn_layer
            L3_cls_DW1_Act["SiLU"]:::act_layer
            L3_cls_C1_Conv["Conv2D: c5 -> c_cls, k1, s1"]:::conv_layer
            L3_cls_C1_BN["BatchNorm2d: c_cls"]:::bn_layer
            L3_cls_C1_Act["SiLU"]:::act_layer
            L3_cls_DW2_Conv["DWConv2D: c_cls -> c_cls, k3, s1"]:::conv_layer
            L3_cls_DW2_BN["BatchNorm2d: c_cls"]:::bn_layer
            L3_cls_DW2_Act["SiLU"]:::act_layer
            L3_cls_C2_Conv["Conv2D: c_cls -> c_cls, k1, s1"]:::conv_layer
            L3_cls_C2_BN["BatchNorm2d: c_cls"]:::bn_layer
            L3_cls_C2_Act["SiLU"]:::act_layer
            
            L3_cls_DW1_Conv --> L3_cls_DW1_BN --> L3_cls_DW1_Act --> L3_cls_C1_Conv --> L3_cls_C1_BN --> L3_cls_C1_Act
            L3_cls_C1_Act --> L3_cls_DW2_Conv --> L3_cls_DW2_BN --> L3_cls_DW2_Act --> L3_cls_C2_Conv --> L3_cls_C2_BN --> L3_cls_C2_Act

            L3_reg_C1_Conv["Conv2D: c5 -> c_reg, k3, s1"]:::conv_layer
            L3_reg_C1_BN["BatchNorm2d: c_reg"]:::bn_layer
            L3_reg_C1_Act["SiLU"]:::act_layer
            L3_reg_C2_Conv["Conv2D: c_reg -> c_reg, k3, s1"]:::conv_layer
            L3_reg_C2_BN["BatchNorm2d: c_reg"]:::bn_layer
            L3_reg_C2_Act["SiLU"]:::act_layer
            
            L3_reg_C1_Conv --> L3_reg_C1_BN --> L3_reg_C1_Act --> L3_reg_C2_Conv --> L3_reg_C2_BN --> L3_reg_C2_Act

            L3_cls_o2m["Conv2D: c_cls -> nc, k1"]:::conv_layer
            L3_reg_o2m["Conv2D: c_reg -> 4*reg_max, k1"]:::conv_layer
            L3_cls_o2o["Conv2D: c_cls -> nc, k1"]:::conv_layer
            L3_reg_o2o["Conv2D: c_reg -> 4*reg_max, k1"]:::conv_layer
            
            L3_cls_C2_Act --> L3_cls_o2m
            L3_cls_C2_Act --> L3_cls_o2o
            L3_reg_C2_Act --> L3_reg_o2m
            L3_reg_C2_Act --> L3_reg_o2o
        end
    end

    %% Kết nối từ Neck sang Detect Head
    N_p3_cv2_Act --> L1_cls_DW1_Conv
    N_p3_cv2_Act --> L1_reg_C1_Conv
    N_n4_cv2_Act --> L2_cls_DW1_Conv
    N_n4_cv2_Act --> L2_reg_C1_Conv
    N_n5_cv2_Act --> L3_cls_DW1_Conv
    N_n5_cv2_Act --> L3_reg_C1_Conv

    %% =========================================================================
    %% 4. HỢP NHẤT VÀ GIẢI MÃ (POST-PROCESSING / BOX DECODING)
    %% =========================================================================
    subgraph PostProcess ["4. Concat & Box Decoding"]
        %% Phân nhánh One-to-Many
        subgraph O2M_Branch ["One-to-Many Branch"]
            O2M_Cls_Flat["Flatten (2) & Transpose"]:::concat_split
            O2M_Reg_Flat["Flatten (2)"]:::concat_split
            O2M_Cls_Concat["Concat: [L1, L2, L3] -> [B, A, nc]"]:::concat_split
            O2M_Reg_Concat["Concat: [L1, L2, L3] -> [B, 4*reg_max, A]"]:::concat_split
            
            %% Giải mã Bounding Box o2m
            subgraph O2M_DFL ["DFL Module"]
                O2M_DFL_Reshape["Reshape -> [B, 4, reg_max, A]"]:::concat_split
                O2M_DFL_Trans["Transpose -> [B, reg_max, 4, A]"]:::concat_split
                O2M_DFL_Softmax["Softmax (dim 1)"]:::act_layer
                O2M_DFL_Conv["Conv2D (1x1, fixed weight=[0,1..,15])"]:::conv_layer
                O2M_DFL_FinalReshape["Reshape -> [B, 4, A]"]:::concat_split
                
                O2M_DFL_Reshape --> O2M_DFL_Trans --> O2M_DFL_Softmax --> O2M_DFL_Conv --> O2M_DFL_FinalReshape
            end
            
            O2M_Box_Dec["Decode: anchors - lt / anchors + rb"]:::concat_split
            O2M_Box_Scale["Scale with Strides & Transpose -> [B, A, 4]"]:::concat_split
        end

        %% Phân nhánh One-to-One
        subgraph O2O_Branch ["One-to-One Branch (NMS-Free Output)"]
            O2O_Cls_Flat["Flatten (2) & Transpose"]:::concat_split
            O2O_Reg_Flat["Flatten (2)"]:::concat_split
            O2O_Cls_Concat["Concat: [L1, L2, L3] -> [B, A, nc]"]:::concat_split
            O2O_Reg_Concat["Concat: [L1, L2, L3] -> [B, 4*reg_max, A]"]:::concat_split
            
            %% Giải mã Bounding Box o2o
            subgraph O2O_DFL ["DFL Module"]
                O2O_DFL_Reshape["Reshape -> [B, 4, reg_max, A]"]:::concat_split
                O2O_DFL_Trans["Transpose -> [B, reg_max, 4, A]"]:::concat_split
                O2O_DFL_Softmax["Softmax (dim 1)"]:::act_layer
                O2O_DFL_Conv["Conv2D (1x1, fixed weight=[0,1..,15])"]:::conv_layer
                O2O_DFL_FinalReshape["Reshape -> [B, 4, A]"]:::concat_split
                
                O2O_DFL_Reshape --> O2O_DFL_Trans --> O2O_DFL_Softmax --> O2O_DFL_Conv --> O2O_DFL_FinalReshape
            end
            
            O2O_Box_Dec["Decode: anchors - lt / anchors + rb"]:::concat_split
            O2O_Box_Scale["Scale with Strides & Transpose -> [B, A, 4]"]:::concat_split
        end

        %% Anchors generator
        AnchorGen["make_anchors: Anchor Points & Stride Tensor"]:::concat_split
    end

    %% Kết nối Predictors L1, L2, L3 sang Concat
    L1_cls_o2m --> O2M_Cls_Flat
    L2_cls_o2m --> O2M_Cls_Flat
    L3_cls_o2m --> O2M_Cls_Flat
    O2M_Cls_Flat --> O2M_Cls_Concat

    L1_reg_o2m --> O2M_Reg_Flat
    L2_reg_o2m --> O2M_Reg_Flat
    L3_reg_o2m --> O2M_Reg_Flat
    O2M_Reg_Flat --> O2M_Reg_Concat

    L1_cls_o2o --> O2O_Cls_Flat
    L2_cls_o2o --> O2O_Cls_Flat
    L3_cls_o2o --> O2O_Cls_Flat
    O2O_Cls_Flat --> O2O_Cls_Concat

    L1_reg_o2o --> O2O_Reg_Flat
    L2_reg_o2o --> O2O_Reg_Flat
    L3_reg_o2o --> O2O_Reg_Flat
    O2O_Reg_Flat --> O2O_Reg_Concat

    %% Flow giải mã hộp o2m
    O2M_Reg_Concat --> O2M_DFL_Reshape
    O2M_DFL_FinalReshape -->|ltrb| O2M_Box_Dec
    AnchorGen -->|anchors| O2M_Box_Dec
    O2M_Box_Dec --> O2M_Box_Scale
    AnchorGen -->|strides| O2M_Box_Scale

    %% Flow giải mã hộp o2o
    O2O_Reg_Concat --> O2O_DFL_Reshape
    O2O_DFL_FinalReshape -->|ltrb| O2O_Box_Dec
    AnchorGen -->|anchors| O2O_Box_Dec
    O2O_Box_Dec --> O2O_Box_Scale
    AnchorGen -->|strides| O2O_Box_Scale

    %% --- ĐẦU RA CUỐI CÙNG ---
    O2M_Cls_Final(["o2m cls output: [B, A, nc]"]):::input_output
    O2M_Box_Final(["o2m box output: [B, A, 4]"]):::input_output
    O2O_Cls_Final(["o2o cls output: [B, A, nc]"]):::input_output
    O2O_Box_Final(["o2o box output: [B, A, 4]"]):::input_output

    O2M_Cls_Concat --> O2M_Cls_Final
    O2M_Box_Scale --> O2M_Box_Final
    O2O_Cls_Concat --> O2O_Cls_Final
    O2O_Box_Scale --> O2O_Box_Final
```

---

## 2. Phân Tích Chi Tiết Các Layer & Khối

### 2.1. Backbone (CSPDarknet-Lite)
Được triển khai trong lớp `Backbone` của file `backbone_neck.py`. Backbone chịu trách nhiệm trích xuất các đặc trưng phân cấp ở 3 mức tỷ lệ (Multi-scale): P3, P4, P5 từ ảnh đầu vào $480 \times 480$ pixel.
* **Lớp Stem:** Tích chập $3 \times 3$ với bước trượt (stride) 2, hạ độ phân giải ảnh xuống $240 \times 240$.
* **Khối C2f (Fast CSP Bottleneck với 2 nhánh):**
  * Tách luồng dữ liệu làm 2 nhánh qua một lớp chập chéo kênh `cv1` ($1 \times 1$ Conv). Nhánh thứ nhất đi thẳng đến Concat cuối khối, nhánh thứ hai đi tuần tự qua chuỗi các khối Bottleneck.
  * Mặc định ở Stage 2 và Stage 3 có `n=2` khối Bottleneck lồng nhau, giúp tăng trường tiếp nhận (Receptive Field) của đặc trưng và tích tụ các liên kết sâu.
  * Các khối Bottleneck ở Backbone sử dụng `shortcut=True` để cộng thêm đặc trưng từ đầu vào, giảm hiện tượng triệt tiêu gradient (vanishing gradient).
* **Khối SPPF (Spatial Pyramid Pooling Fast):**
  * Đặt ở cuối Stage 4 (tầng P5), có tác dụng trích xuất đặc trưng ngữ nghĩa toàn cục cực kỳ hiệu quả bằng cách song song hóa các phép Max Pooling kích thước $5 \times 5$ xếp chồng. Sau đó concat toàn bộ luồng lại và nén kênh qua một lớp Conv $1 \times 1$.

### 2.2. Neck (PAFPN)
Được triển khai trong lớp `PAFPN` của file `backbone_neck.py`. Neck là kiến trúc **Path Aggregation Network** kết hợp luồng đi từ trên xuống (Top-Down) và từ dưới lên (Bottom-Up):
* **Đường đi Top-Down (FPN):**
  * Đặc trưng ngữ nghĩa mức cao của tầng P5 được làm mịn bằng lớp Conv $1 \times 1$ (`reduce5`), rồi được nội suy phóng đại kích thước gấp 2 lần (`Upsample`) để nối (Concatenate) với đặc trưng của tầng P4.
  * Hỗn hợp này đi qua khối `C2f` không chứa shortcut (`shortcut=False`) để học quan hệ chéo kênh mà không giữ nguyên nhiễu cục bộ. Quá trình tương tự diễn ra khi truyền từ P4 xuống P3.
* **Đường đi Bottom-Up (PAN):**
  * Sau khi đi qua nhánh Top-Down, luồng thông tin định vị chính xác ở mức thấp (từ P3) sẽ được chập stride 2 (`down3` và `down4`) để giảm chiều không gian và Concatenate ngược lên các tầng P4, P5 phía trên.
  * Sự dung hợp hai chiều giúp các đầu ra cuối cùng của Neck ($P3_{out}$, $P4_{out}$, $P5_{out}$) vừa giàu thông tin ngữ nghĩa trừu tượng vừa chính xác về mặt định vị không gian.

### 2.3. Detect Head & Dual Branches
Đầu phát hiện được triển khai trong lớp `DetectHead` của file `head.py`. Đây là nơi hiện thực hóa cơ chế **Dual Label Assignment** của YOLOv10:
* **Tách luồng đặc trưng (Decoupled Head):**
  * Với mỗi mức đặc trưng đầu vào, thông tin được rẽ thành 2 nhánh:
    * **Classification Stem (`cls_stem`):** Gồm 2 khối chập Depthwise Separable (`DWConv`) xen kẽ với 2 khối chập thường ($1\times1$) giúp giảm đáng kể lượng tham số tính toán nhưng vẫn giữ được khả năng học phân lớp đa dạng.
    * **Regression Stem (`reg_stem`):** Gồm 2 khối chập thường $3 \times 3$ nối tiếp giúp bảo toàn tối đa độ nhạy và chính xác hình học cần thiết cho hồi quy tọa độ.
* **Phân nhánh dự đoán song song:**
  * **One-to-Many predictor (`o2m`):** Đầu ra dự đoán phân loại (`cls_o2m`) và hồi quy (`reg_o2m`). Được giám sát trong lúc huấn luyện với $K=10$ anchors tốt nhất cho mỗi Ground Truth nhằm cung cấp luồng gradient phong phú giúp Backbone/Neck học đặc trưng tối ưu.
  * **One-to-One predictor (`o2o`):** Đầu ra dự đoán độc lập (`cls_o2o`) và hồi quy (`reg_o2o`). Được giám sát với $K=1$ anchor duy nhất. Nhánh này ép mô hình tự loại bỏ các dự đoán trùng lặp.
  * **Inference Mode:** Khi suy luận thực tế, ta hoàn toàn có thể bỏ qua nhánh `o2m` và chỉ lấy trực tiếp đầu ra từ nhánh `o2o`, giúp suy luận trực tiếp **không cần thuật toán lọc NMS** truyền thống, loại bỏ trễ thời gian thực.

### 2.4. Phân phối hồi quy và giải mã (DFL Decoding)
* Đầu ra hồi quy của mỗi nhánh có kích thước kênh là $4 \times reg\_max$ (mặc định $4 \times 16 = 64$ kênh).
* Lớp `DFL` (`blocks.py`) thực hiện chia nhỏ kênh này thành 4 nhóm tương ứng 4 cạnh biên (left, top, right, bottom), mỗi nhóm 16 kênh.
* Một hàm `Softmax` được áp dụng trên chiều 16 kênh của từng nhóm để sinh ra một phân phối xác suất rời rạc.
* Lớp chập $1 \times 1$ không huấn luyện nhân phân phối này với vector trọng số tăng dần từ `[0, 1, 2, ..., 15]`, thu được kỳ vọng toán học đại diện cho khoảng cách thực tế của box tới tâm điểm neo.
* Cuối cùng, hàm `decode_box` trong `DetectHead` cộng/trừ các khoảng cách này với tọa độ điểm neo (`anchors`) và nhân với bước nhảy (`strides`) của tầng tương ứng để quy đổi ra tọa độ góc dạng pixel `[x1, y1, x2, y2]`.
