
import sys
try:
    from pypdf import PdfReader
except ImportError:
    try:
        import PyPDF2 as PdfReader
    except ImportError:
        print("No PDF library found")
        sys.exit(1)

pdf_path = r"C:\Users\rdaksh\Desktop\Agri AI\Prithvi\irjmets71200064198.pdf"
try:
    print("Analyzing PDF content...")
    reader = PdfReader(pdf_path)  # Re-initialize reader
    full_text = ""
    for page in reader.pages: # Read all pages
        full_text += page.extract_text() + "\n"
    
    keywords = ["sugarcane", "rice", "sentinel", "prithvi", "transformer", "segmentation", "vi", "ndvi"]
    found = []
    
    lower_text = full_text.lower()
    for k in keywords:
        if k in lower_text:
            count = lower_text.count(k)
            found.append(f"{k} ({count})")
            
    print(f"Keywords found: {', '.join(found)}")
    print("\nSummary/Abstract snippet:")
    print(full_text[:500]) # First 500 chars
except Exception as e:
    print(f"Error reading PDF: {e}")
