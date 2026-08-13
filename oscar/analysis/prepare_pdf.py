import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def save_plots_to_pdf(
    figs,
    captions,                 # list[str] (same length as figs) OR a single str for all
    pdf_path,
    metadata=None,
    caption_y=0.02,               # bottom margin position (figure coords)
    fontsize=9,
    pad_bottom=0.20,              # extra bottom room for caption
    box=True
):
    print(pdf_path)
    
    if isinstance(captions, str):
        captions = [captions] * len(figs)
    if len(captions) != len(figs):
        raise ValueError("The number of captions must match number of figures (or be a single string).")

    base, ext = os.path.splitext(pdf_path)
    tmp = f"{base}.tmp{ext}"

    with PdfPages(tmp) as pdf:
        if metadata:
            pdf.infodict().update(metadata)

        for i, (fig, expl) in enumerate(zip(figs, captions), start=1):
            # Make space for caption (don’t shrink if user already set a larger bottom)
            fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, pad_bottom))

            # Add numbered caption
            text = f"Figure {i}: {expl}"
            bbox = dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9) if box else None
            caption_artist = fig.text(
                0.5, caption_y, text,
                ha="center", va="bottom",
                fontsize=fontsize,
                wrap=True,
                bbox=bbox,
            )

            pdf.savefig(fig)
            # Clean up so original figs remain unmodified
            caption_artist.remove()

    os.replace(tmp, pdf_path)
    print('\n\n************************************************')
    print('All DONE')
