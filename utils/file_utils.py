import os
import json
from config.constants import ROOT_DIR, IMAGES_OUTPUT_DIR, JSON_OUTPUT_DIR

def getNextTestNumber():
	"""Get the next available test number"""
	existing_files = []
	
	# Check both images and json directories for existing files
	for directory in [IMAGES_OUTPUT_DIR, JSON_OUTPUT_DIR]:
		if os.path.exists(directory):
			for f in os.listdir(directory):
				if f.startswith(('plan', 'vis', 'acc', 'final')) and any(f.endswith(ext) for ext in ['.json', '.png']):
					existing_files.append(f)
	
	# Also check root directory for backward compatibility
	for f in os.listdir(ROOT_DIR):
		if f.startswith(('plan', 'vis', 'acc', 'final')) and any(f.endswith(ext) for ext in ['.json', '.png']):
			existing_files.append(f)
	
	# Find highest test number
	max_num = 0
	for f in existing_files:
		try:
			# Extract number from names like plan1.json, vis1.png, acc1.json, final1.json
			if 'plan' in f or 'vis' in f or 'acc' in f or 'final' in f:
				import re
				match = re.search(r'(\d+)', f)
				if match:
					num = int(match.group(1))
					max_num = max(max_num, num)
		except:
			pass
	
	return max_num + 1

def saveJsonToFile(json_data, custom_name=None):
	"""Save JSON data to file in JSON output directory with simple naming"""
	if custom_name:
		filename = f"{custom_name}.json"
	else:
		test_num = getNextTestNumber()
		filename = f"plan{test_num}.json"
	
	filepath = os.path.join(JSON_OUTPUT_DIR, filename)
	
	try:
		with open(filepath, 'w') as f:
			json.dump(json_data, f, indent=2)
		print(f"JSON saved to: {filepath}")
		return filename
	except Exception as e:
		print(f"Error saving JSON file: {str(e)}")
		return None

def saveAccuracyAnalysis(accuracy_data, test_num):
	"""Save accuracy analysis to file with simple naming"""
	filename = f"acc{test_num}.json"
	filepath = os.path.join(JSON_OUTPUT_DIR, filename)
	
	try:
		with open(filepath, 'w') as f:
			json.dump(accuracy_data, f, indent=2)
		print(f"Accuracy analysis saved to: {filepath}")
		return filename
	except Exception as e:
		print(f"Error saving accuracy analysis: {str(e)}")
		return None