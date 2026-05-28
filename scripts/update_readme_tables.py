import os
import pandas as pd
import numpy as np
import re

def generate_markdown_tables(output_dir):
    # Load all CSVs
    pca_recon = pd.read_csv(os.path.join(output_dir, "pca_reconstruction_error.csv"))
    pca_dir = pd.read_csv(os.path.join(output_dir, "pca_direction_score.csv"))
    pca_score = pd.read_csv(os.path.join(output_dir, "pca_score.csv"))
    kmeans_score = pd.read_csv(os.path.join(output_dir, "kmeans_score.csv"))
    dbscan_score = pd.read_csv(os.path.join(output_dir, "dbscan_score.csv"))
    entropy_score = pd.read_csv(os.path.join(output_dir, "entropy_score.csv"))
    hybrid_score = pd.read_csv(os.path.join(output_dir, "hybrid_rrf_score.csv"))

    tables = {}

    # Helper function to get emoji prefix for top ranks
    def rank_emoji(r):
        if r == 1: return "🥇 1"
        elif r == 2: return "🥈 2"
        elif r == 3: return "🥉 3"
        return str(r)

    def rank_emoji_suffix(r):
        if r == 1: return "🥇 1st"
        elif r == 2: return "🥈 2nd"
        elif r == 3: return "🥉 3rd"
        elif r == 4: return "4th"
        elif r == 5: return "5th"
        elif r == 6: return "6th"
        elif r == 7: return "7th"
        elif r == 8: return "8th"
        elif r == 9: return "9th"
        return f"{r}th"

    # 1. PCA Reconstruction Error Table (sorted by feature name or original order? In original order)
    # Let's keep original order as in the file
    t1 = ["| Feature | Recon Score (E_i − E_full) |", "|---|---|"]
    for _, row in pca_recon.iterrows():
        val = row['scores']
        val_str = f"+{val:.6f}" if val >= 0 else f"−{abs(val):.6f}"
        t1.append(f"| {row['features']} | {val_str} |")
    tables["pca_reconstruction_error"] = "\n".join(t1)

    # 2. PCA Direction Score Table (sorted descending)
    t2_df = pca_dir.sort_values(by="scores", ascending=False).reset_index(drop=True)
    t2 = ["| Feature | Direction Score (1 − cos_sim) | Rank |", "|---|---|---|"]
    for idx, row in t2_df.iterrows():
        rank_str = rank_emoji_suffix(idx + 1)
        t2.append(f"| **{row['features']}** | **{row['scores']:.4f}** | {rank_str} |" if idx < 2 else f"| {row['features']} | {row['scores']:.4f} | {rank_str} |")
    tables["pca_direction_score"] = "\n".join(t2)

    # 3. Entropy Score Table (sorted descending)
    t3_df = entropy_score.sort_values(by="scores", ascending=False).reset_index(drop=True)
    t3 = ["| Feature | Entropy H(X_m) |", "|---|---|"]
    for _, row in t3_df.iterrows():
        t3.append(f"| {row['features']} | {row['scores']:.4f} |")
    tables["entropy_score"] = "\n".join(t3)

    # 4. PCA Score Table (sorted descending)
    t4_df = pca_score.sort_values(by="scores", ascending=False).reset_index(drop=True)
    t4 = ["| Feature | PCA Score | Rank |", "|---|---|---|"]
    for idx, row in t4_df.iterrows():
        rank_str = rank_emoji_suffix(idx + 1)
        t4.append(f"| **{row['features']}** | **{row['scores']:.4f}** | {rank_str} |" if idx < 2 else f"| {row['features']} | {row['scores']:.4f} | {rank_str} |")
    tables["pca_score"] = "\n".join(t4)

    # 5. KMeans Score Table (sorted descending)
    t5_df = kmeans_score.sort_values(by="scores", ascending=False).reset_index(drop=True)
    t5 = ["| Feature | ΔSilhouette Score | Signal |", "|---|---|---|"]
    for idx, row in t5_df.iterrows():
        sig = "🥇 Most helpful" if idx == 0 else ("🥈 2nd helpful" if idx == 1 else ("🔻 Most disruptive" if idx == len(t5_df)-1 else ("hurts cluster quality" if row['scores'] < -0.019 else "slightly hurts")))
        # customize based on values
        val = row['scores']
        val_str = f"+{val:.6f}" if val >= 0 else f"−{abs(val):.6f}"
        t5.append(f"| **{row['features']}** | **{val_str}** | {sig} |" if idx < 2 or idx == len(t5_df)-1 else f"| {row['features']} | {val_str} | {sig} |")
    tables["kmeans_score"] = "\n".join(t5)

    # 6. DBSCAN Score Table (sorted descending)
    t6_df = dbscan_score.sort_values(by="scores", ascending=False).reset_index(drop=True)
    t6 = ["| Feature | DBSCAN ΔSilhouette | Signal |", "|---|---|---|"]
    # DBSCAN often has ties for top score. Find unique max values
    max_score = t6_df['scores'].max()
    min_score = t6_df['scores'].min()
    for idx, row in t6_df.iterrows():
        val = row['scores']
        val_str = f"+{val:.6f}" if val >= 0 else f"−{abs(val):.6f}"
        if abs(val - max_score) < 1e-7:
            sig = "🥇 Critical (tied)"
            bold_feat = f"**{row['features']}**"
            bold_val = f"**{val_str}**"
        elif abs(val - min_score) < 1e-7:
            sig = "🔻 Most disruptive"
            bold_feat = f"**{row['features']}**"
            bold_val = f"**{val_str}**"
        else:
            sig = "minor negative" if val > -0.01 else "moderate negative"
            bold_feat = row['features']
            bold_val = val_str
        t6.append(f"| {bold_feat} | {bold_val} | {sig} |")
    tables["dbscan_score"] = "\n".join(t6)

    # 7. Strategy Comparison Table
    # We need ranks for PCA, KMeans, DBSCAN, and Entropy
    feat_ranks = {}
    for f in pca_score['features']:
        feat_ranks[f] = {}

    # Helper to populate rank map
    def fill_ranks(df, key):
        sorted_df = df.sort_values(by="scores", ascending=False).reset_index(drop=True)
        for idx, row in sorted_df.iterrows():
            feat_ranks[row['features']][key] = idx + 1

    fill_ranks(pca_score, 'pca')
    fill_ranks(kmeans_score, 'kmeans')
    fill_ranks(dbscan_score, 'dbscan')
    fill_ranks(entropy_score, 'entropy')

    # Sort the comparison table features by their PCA rank for consistent display
    comp_features = sorted(feat_ranks.keys(), key=lambda x: feat_ranks[x]['pca'])
    t7 = [
        "| Feature | PCA Rank | KMeans Rank | DBSCAN Rank | Entropy Rank |",
        "|---|---|---|---|---|",
    ]
    for f in comp_features:
        r_pca = feat_ranks[f]['pca']
        r_kmeans = feat_ranks[f]['kmeans']
        r_dbscan = feat_ranks[f]['dbscan']
        r_entropy = feat_ranks[f]['entropy']
        
        suffix = ""
        if r_kmeans == 1 and r_dbscan == 1:
            suffix = "  ← consistent top"
        t7.append(f"| {f} | {r_pca} | {r_kmeans} | {r_dbscan} | {r_entropy}{suffix} |")
    tables["strategy_comparison"] = "\n".join(t7)

    # 8. Hybrid RRF Score Table
    # Sort hybrid score descending
    t8_df = hybrid_score.sort_values(by="HybridScore", ascending=False).reset_index(drop=True)
    t8 = ["| Final Rank | Feature | Hybrid RRF Score |", "|---|---|---|"]
    for idx, row in t8_df.iterrows():
        rank_str = rank_emoji(idx + 1)
        feat_str = f"**{row['features']}**" if idx < 3 else row['features']
        score_str = f"**{row['HybridScore']:.6f}**" if idx < 3 else f"{row['HybridScore']:.6f}"
        t8.append(f"| {rank_str} | {feat_str} | {score_str} |")
    tables["hybrid_rrf_score"] = "\n".join(t8)

    # 9. Final RRF Output Table
    # Re-use the sorted hybrid score, showing the top ranks
    t9 = ["| Rank | Feature | Hybrid Score |", "|---|---|---|"]
    for idx, row in t8_df.iterrows():
        t9.append(f"| {idx + 1} | {row['features']} | {row['HybridScore']:.6f} |")
    tables["final_rrf_output"] = "\n".join(t9)

    return tables

def update_file(file_path, tables):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
        
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    updated = content
    for name, table_md in tables.items():
        pattern = re.compile(
            rf"(<!--\s*START_TABLE:\s*{name}\s*-->).*?(<!--\s*END_TABLE:\s*{name}\s*-->)",
            re.DOTALL
        )
        if pattern.search(updated):
            updated = pattern.sub(rf"\1\n{table_md}\n\2", updated)
            print(f"Updated table: {name} in {file_path}")
        else:
            print(f"Warning: Table placeholder for '{name}' not found in {file_path}")

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(updated)

def main():
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(workspace_dir, "notebook", "output")
    readme_path = os.path.join(workspace_dir, "README.md")
    notebook_readme_path = os.path.join(workspace_dir, "notebook", "amf_re.md")

    print(f"Generating tables from outputs in: {output_dir}")
    tables = generate_markdown_tables(output_dir)

    print("Updating README.md...")
    update_file(readme_path, tables)

    print("Updating notebook/amf_re.md...")
    update_file(notebook_readme_path, tables)

if __name__ == "__main__":
    main()
