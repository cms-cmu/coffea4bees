#!/usr/bin/env python3
"""
Merge and plot histograms across year folders in a ROOT file.

This script opens a ROOT file, looks for the same histogram inside three (or
configurable) folders like HH4b_2016, HH4b_2017, HH4b_2018, merges them (sum),
and plots the result on one canvas. You can pass many histogram names; the
script will create one output image per name.

Requirements:
- PyROOT (ROOT Python bindings) available in your environment.
- Optionally, a 'cmsstyle' module for styling; otherwise a TLatex label is used.

Example usage:
  python test_vbf.py \
	--root-file /path/to/file.root \
	--hists qqHH_CV_m1p21_C2V_1p94_kl_m0p94_13p0TeV_hbbhbb mass_bb1 \
	--outdir plots --formats png pdf --logy --label-left CMS --label-right Preliminary

Notes:
- If a histogram is missing in one of the folders, the script will keep going
  and sum the ones found. If none are found, it will warn and skip that name.
- Output file names are sanitized (non-alphanumeric replaced by '_').
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import List, Optional

try:
	import ROOT  # type: ignore
except Exception as e:  # pragma: no cover
	print("FATAL: Failed to import ROOT. Ensure PyROOT is available in your environment.")
	print(e)
	sys.exit(1)


def try_import_cmsstyle():
	"""Try to import cmsstyle (optional). Return module or None."""
	try:
		import cmsstyle as _cmsstyle  # type: ignore

		return _cmsstyle
	except Exception:
		return None


def sanitize_filename(name: str) -> str:
	# Replace any non-word character with underscore and collapse repeats
	s = re.sub(r"\W+", "_", name.strip())
	return s.strip("_") or "hist"


def get_hist(tfile: "ROOT.TFile", folder: str, hist_name: str):
	"""Fetch histogram at folder/hist_name from tfile, return None if not found or not TH1."""
	obj = tfile.Get(f"{folder}/{hist_name}")
	if not obj:
		return None
	# Accept TH1 or derivatives
	if not isinstance(obj, ROOT.TH1):
		# Some ROOT versions won't make isinstance work as expected; fallback to InheritsFrom
		if not obj.InheritsFrom("TH1"):
			return None
	return obj


def sum_hists(hists: List["ROOT.TH1"], name: str) -> Optional["ROOT.TH1"]:
	if not hists:
		return None
	merged = hists[0].Clone(f"{name}_merged")
	merged.SetDirectory(0)
	# Ensure proper error treatment
	merged.Sumw2()
	for h in hists[1:]:
		merged.Add(h)
	return merged


def draw_label(label_left: str = "CMS", label_right: str = "Preliminary", extra: Optional[str] = None):
	latex = ROOT.TLatex()
	latex.SetNDC(True)
	latex.SetTextFont(62)  # CMS bold
	latex.SetTextSize(0.045)
	latex.DrawLatex(0.14, 0.88, label_left)
	if label_right:
		latex2 = ROOT.TLatex()
		latex2.SetNDC(True)
		latex2.SetTextFont(52)  # supplementary
		latex2.SetTextSize(0.035)
		latex2.DrawLatex(0.245, 0.88, label_right)
	if extra:
		latex3 = ROOT.TLatex()
		latex3.SetNDC(True)
		latex3.SetTextFont(42)
		latex3.SetTextSize(0.035)
		latex3.DrawLatex(0.14, 0.84, extra)


def apply_basic_style():
	ROOT.gROOT.SetBatch(True)
	ROOT.gStyle.SetOptStat(0)
	ROOT.gStyle.SetTitleBorderSize(0)
	ROOT.gStyle.SetPadTopMargin(0.08)
	ROOT.gStyle.SetPadRightMargin(0.05)
	ROOT.gStyle.SetPadBottomMargin(0.12)
	ROOT.gStyle.SetPadLeftMargin(0.12)


def plot_hist(hist: "ROOT.TH1", title: Optional[str], outpaths: List[str], logy: bool, legend_note: Optional[str], 
			  label_left: str, label_right: str, extra_label: Optional[str]):
	c = ROOT.TCanvas("c", "c", 800, 700)
	if logy:
		c.SetLogy(True)
	# Style
	hist.SetLineColor(ROOT.kAzure + 1)
	hist.SetLineWidth(2)
	hist.SetFillColorAlpha(ROOT.kAzure - 9, 0.35)
	hist.SetTitle(title or hist.GetTitle())
	# Axis label sizes
	hist.GetXaxis().SetTitleSize(0.045)
	hist.GetYaxis().SetTitleSize(0.045)
	hist.GetXaxis().SetLabelSize(0.04)
	hist.GetYaxis().SetLabelSize(0.04)
	hist.SetMinimum(0)
	# Reasonable y-range for log
	if logy:
		min_val = 0.5 * min([x for x in [hist.GetMinimum(1e-9)] if x is not None] + [1e-6])
		hist.SetMinimum(max(min_val, 1e-6))
		hist.SetMaximum(hist.GetMaximum() * 50.0)

	hist.Draw("HISTE")

	# Legend
	if legend_note:
		leg = ROOT.TLegend(0.55, 0.78, 0.9, 0.9)
		leg.SetBorderSize(0)
		leg.SetFillStyle(0)
		leg.AddEntry(hist, legend_note, "f")
		leg.Draw()

	draw_label(label_left, label_right, extra_label)

	for path in outpaths:
		c.SaveAs(path)

	# Prevent canvas from being garbage collected too early if running interactively
	c.Close()


def main(argv: Optional[List[str]] = None) -> int:
	parser = argparse.ArgumentParser(description="Merge and plot histograms across year folders in a ROOT file.")
	parser.add_argument("--root-file", required=True, help="Path to the input ROOT file")
	parser.add_argument("--hists", nargs='+', required=True, help="One or more histogram names to merge")
	parser.add_argument("--folders", nargs='+', default=["HH4b_2016", "HH4b_2017", "HH4b_2018"],
						help="Folders (TDirectory) to search within the ROOT file")
	parser.add_argument("--outdir", default=".", help="Directory to save plots")
	parser.add_argument("--formats", nargs='+', default=["png"], help="Output image formats, e.g. png pdf root")
	parser.add_argument("--logy", action="store_true", help="Use logarithmic y-axis")
	parser.add_argument("--title", default=None, help="Override histogram title on the canvas")
	parser.add_argument("--legend-note", default=None, help="Legend note to describe the plot")
	parser.add_argument("--label-left", default="CMS", help="Left label text (e.g., 'CMS')")
	parser.add_argument("--label-right", default="Preliminary", help="Right label text (e.g., 'Preliminary')")
	parser.add_argument("--extra-label", default=None, help="Extra label under CMS, e.g., 'Run 2 (2016-2018)'")
	parser.add_argument("--try-cmsstyle", action="store_true", help="Attempt to import and apply cmsstyle if available")

	args = parser.parse_args(argv)

	apply_basic_style()

	if args.try_cmsstyle:
		cmsstyle = try_import_cmsstyle()
		if cmsstyle:
			# Best-effort: try a few common entry points; ignore failures silently
			for attr in ("setCMSTDRStyle", "setTDRStyle", "tdrstyle"):
				try:
					fn = getattr(cmsstyle, attr)
					if callable(fn):
						fn()
						break
				except Exception:
					pass

	infile = args.root_file
	if not os.path.isfile(infile):
		print(f"ERROR: ROOT file not found: {infile}")
		return 2

	os.makedirs(args.outdir, exist_ok=True)

	tfile = ROOT.TFile.Open(infile)
	if not tfile or tfile.IsZombie():
		print(f"ERROR: Could not open ROOT file: {infile}")
		return 2

	exit_code = 0
	for hist_name in args.hists:
		found_hists: List[ROOT.TH1] = []
		missing_in: List[str] = []
		folder_hist_map = {}  # Track which histogram came from which folder
		
		for folder in args.folders:
			h = get_hist(tfile, folder, hist_name)
			if h:
				found_hists.append(h)
				folder_hist_map[folder] = h
			else:
				missing_in.append(folder)

		if not found_hists:
			print(f"WARNING: No histograms found for '{hist_name}' in any of: {', '.join(args.folders)}. Skipping.")
			exit_code = max(exit_code, 1)
			continue

		if missing_in:
			print(f"Note: '{hist_name}' missing in: {', '.join(missing_in)}. Proceeding with available ones.")

		base = sanitize_filename(hist_name)
		
		# Plot individual year histograms
		for folder, h in folder_hist_map.items():
			# Extract year suffix from folder name (e.g., "2016" from "HH4b_2016")
			year_suffix = folder.split('_')[-1] if '_' in folder else folder
			outpaths = [os.path.join(args.outdir, f"{base}_{year_suffix}.{ext}") for ext in args.formats]
			legend_note = args.legend_note or folder
			plot_hist(h, args.title, outpaths, args.logy, legend_note, args.label_left, args.label_right, args.extra_label)
			print(f"Saved: {', '.join(outpaths)}")

		# Plot merged histogram
		merged = sum_hists(found_hists, hist_name)
		if not merged:
			print(f"WARNING: Failed to merge histogram '{hist_name}'. Skipping.")
			exit_code = max(exit_code, 1)
			continue

		outpaths = [os.path.join(args.outdir, f"{base}_merged.{ext}") for ext in args.formats]
		legend_note = args.legend_note or f"Sum of {', '.join(folder_hist_map.keys())}"
		plot_hist(merged, args.title, outpaths, args.logy, legend_note, args.label_left, args.label_right, args.extra_label)
		print(f"Saved: {', '.join(outpaths)}")

	tfile.Close()
	return exit_code


if __name__ == "__main__":
	sys.exit(main())

