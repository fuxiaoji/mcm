# PCA 3D Visualization: Contestant Profiles Analysis

## Overview

This document describes the Principal Component Analysis (PCA) visualization of DWTS contestant profiles, providing a dimensionality reduction view of multi-dimensional contestant characteristics.

---

## 1. Data Source

**File**: `问题1_选手汇总表.csv`  
**Sample Size**: 384 contestants (with complete data)

### Features Used for PCA

| Feature | Description | Unit |
|---------|-------------|------|
| `celebrity_age` | Age of celebrity during season | Years |
| `weeks_survived` | Number of weeks survived in competition | Weeks (1-11) |
| `season_avg_score` | Average judge score across all weeks | Points (1-10) |
| `fan_vote_estimate` | MCMC-estimated fan vote share | Proportion (0-1) |
| `placement` | Final placement in the competition | Rank (1 = Champion) |

---

## 2. PCA Results Summary

### Variance Explained

| Component | Variance Explained | Cumulative |
|-----------|-------------------|------------|
| **PC1** | 70.04% | 70.04% |
| **PC2** | 15.38% | 85.42% |
| **PC3** | 9.41% | **94.83%** |

> Three principal components capture **94.83%** of total variance.

### Loadings Matrix

| Feature | PC1 | PC2 | PC3 |
|---------|-----|-----|-----|
| `celebrity_age` | +0.316 | **+0.889** | -0.281 |
| `weeks_survived` | **-0.495** | +0.174 | -0.342 |
| `season_avg_score` | **-0.482** | -0.099 | -0.364 |
| `fan_vote_estimate` | -0.407 | +0.370 | **+0.807** |
| `placement` | **+0.506** | -0.181 | +0.143 |

---

## 3. Principal Component Interpretation

### PC1 (70.04%): "Competition Performance Index"

**Interpretation**: Measures overall competition success

| Direction | Characteristics |
|-----------|-----------------|
| **PC1 > 0** (Positive) | Early elimination, poor placement, low scores |
| **PC1 < 0** (Negative) | Late survival, good placement, high scores |

**Key Contributors**:
- `placement` (+0.506): Higher rank number → Higher PC1
- `weeks_survived` (-0.495): More weeks → Lower PC1
- `season_avg_score` (-0.482): Higher scores → Lower PC1

**In Plain Language**: PC1 is essentially a "poor performance" index. Contestants with high PC1 values were eliminated early with low scores; contestants with low PC1 values performed well throughout.

---

### PC2 (15.38%): "Age-Popularity Axis"

**Interpretation**: Captures the age dimension and its correlation with fan support

| Direction | Characteristics |
|-----------|-----------------|
| **PC2 > 0** (Positive) | Older celebrities with stronger fan bases |
| **PC2 < 0** (Negative) | Younger contestants |

**Key Contributors**:
- `celebrity_age` (+0.889): Dominant loading - older → higher PC2
- `fan_vote_estimate` (+0.370): More fan votes → higher PC2

**In Plain Language**: PC2 distinguishes contestants by age. Older celebrities (often with established fan bases from their original careers) cluster at high PC2 values.

---

### PC3 (9.41%): "Fan-Driven vs. Skill-Driven Success"

**Interpretation**: Separates contestants who rely on fan votes vs. technical skill

| Direction | Characteristics |
|-----------|-----------------|
| **PC3 > 0** (Positive) | High fan votes, moderate technical scores |
| **PC3 < 0** (Negative) | High technical scores, moderate fan votes |

**Key Contributors**:
- `fan_vote_estimate` (+0.807): Dominant loading - more fan votes → higher PC3
- `season_avg_score` (-0.364): Higher scores → lower PC3

**In Plain Language**: PC3 captures the "controversy" dimension. Contestants like Bobby Bones (high PC3) won through fan support despite lower scores, while technical dancers like Jennifer Grey (low PC3) won through skill.

---

## 4. Grouping Analysis: Season Era

### Groups Defined

| Era | Seasons | Sample Size | Color |
|-----|---------|-------------|-------|
| Early | S1-10 | 110 | 🔵 Blue |
| Middle | S11-20 | 105 | 🔴 Red |
| Recent | S21-33 | 169 | 🟢 Green |

### Key Findings

1. **High Overlap Between Groups**: The three era groups show substantial overlap in PCA space, suggesting contestant profiles have remained relatively consistent across DWTS history.

2. **No Dramatic Era Shift**: Unlike the reference figure (showing distinct year clusters), DWTS contestant characteristics have not fundamentally changed over 33 seasons.

3. **This IS a Finding**: The overlap itself is meaningful - it demonstrates the show has maintained consistent casting patterns (age distribution, industry mix, skill levels) throughout its run.

---

## 5. Figure Descriptions

### Figure: pca_3d_season_era.png

**Title**: Principal Component Analysis: Contestant Profiles by Season Era

**Panel Description**:
- **3D scatter plot** showing all 384 contestants in PC1 × PC2 × PC3 space
- **Color coding**: Blue (Early), Red (Middle), Green (Recent)
- **95% confidence ellipsoids** for each era group
- **Axis labels** include variance explained percentages

**Usage in Paper**: This figure can be used in Section 2 (Data Preprocessing) or Section 5 (Model Analysis) to demonstrate the multi-dimensional structure of contestant data.

---

### Figure: pca_2d_season_era.png

**Title**: PCA: Contestant Profiles by Season Era (2D Projection)

**Panel Description**:
- **2D scatter plot** showing PC1 vs PC2 projection
- Same color coding and confidence ellipses as 3D version
- Clearer visualization for print/PDF format

**Usage in Paper**: Preferred for paper submission due to clearer 2D representation.

---

## 6. LaTeX Figure Code

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.9\textwidth]{figures/pca_2d_season_era.png}
    \caption{Principal Component Analysis of contestant profiles. 
    PC1 (70.0\%) captures overall competition performance, 
    PC2 (15.4\%) reflects age demographics, and 
    PC3 (9.4\%) distinguishes fan-driven vs. skill-driven success. 
    The substantial overlap between season eras indicates consistent 
    casting patterns throughout DWTS history. 
    Ellipses represent 95\% confidence regions.}
    \label{fig:pca_analysis}
\end{figure}
```

---

## 7. Files in This Folder

| File | Description |
|------|-------------|
| `pca_3d_season_era.png` | 3D PCA scatter plot with confidence ellipsoids |
| `pca_2d_season_era.png` | 2D projection (PC1 vs PC2) for paper |
| `pca_3d_industry.png` | Alternative grouping by celebrity industry |
| `pca_2d_industry.png` | 2D version of industry grouping |
| `README.md` | This documentation file |

---

## 8. Code Reference

**Script**: `/Users/Zhuanz1/Desktop/mcm/c/论文图表/pca_3d_v2.py`

To regenerate figures:
```bash
cd /Users/Zhuanz1/Desktop/mcm/c/论文图表
python pca_3d_v2.py
```

---

## 9. Summary for Paper

> **Suggested Text for Paper**:
> 
> We applied Principal Component Analysis to reduce the five-dimensional contestant feature space (age, weeks survived, average score, fan vote estimate, and final placement) to three principal components explaining 94.83% of total variance. PC1 (70.0%) captures overall competition performance, with high values indicating early elimination and low scores. PC2 (15.4%) primarily reflects contestant age demographics. PC3 (9.4%) distinguishes between fan-driven and skill-driven success patterns. 
>
> Visualization of contestants grouped by season era (Early: S1-10, Middle: S11-20, Recent: S21-33) reveals substantial overlap in PCA space, indicating that DWTS has maintained consistent contestant profiles throughout its 33-season history. This stability suggests that observed differences in competition outcomes are more attributable to scoring methodology changes than to shifts in contestant demographics.
