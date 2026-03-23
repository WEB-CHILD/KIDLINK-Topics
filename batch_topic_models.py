#!/usr/bin/env python3
"""
Batch Topic Model Generator for Workshop Participants

Reads a configuration CSV with attendee information and generates topic models
for each participant's data across multiple domains.

Configuration CSV format:
    att,domain,csv-name
    attendee1,domain1,data_file1.csv
    attendee1,domain2,data_file2.csv
    attendee2,domain1,data_file3.csv
    ...

Usage:
    python batch_topic_models.py <config_csv_file>
"""

import csv
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json
import logging


def setup_logging():
    """Configure logging to both file and console."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"batch_processing_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return log_file


def validate_inputs(config_csv: Path, rows: list) -> bool:
    """
    Validate that all required input files exist.
    Marks rows with '_skip' flag if they're too small.
    """
    logger = logging.getLogger()
    all_valid = True
    MIN_ROWS = 150  # Minimum data rows required for UMAP dimensionality reduction
    
    for i, row in enumerate(rows):
        att = row.get("att", "").strip()
        domain = row.get("domain", "").strip()
        csv_name = row.get("csv-name", "").strip()
        
        # Validate required fields
        if not att or not domain or not csv_name:
            logger.error(f"Row {i+2}: Missing required field(s). att={att}, domain={domain}, csv-name={csv_name}")
            all_valid = False
            row["_skip"] = True
            continue
        
        # Check if input CSV exists
        csv_path = Path("data") / csv_name
        if not csv_path.exists():
            logger.error(f"Row {i+2}: Input CSV not found: {csv_path}")
            all_valid = False
            row["_skip"] = True
            continue
        
        # Check CSV size (count rows)
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                row_count = sum(1 for _ in f) - 1  # -1 for header
            
            if row_count < MIN_ROWS:
                logger.warning(f"Row {i+2}: Skipping {att}_{domain} - only {row_count} documents (minimum: {MIN_ROWS})")
                row["_skip"] = True
            else:
                row["_skip"] = False
        except Exception as e:
            logger.error(f"Row {i+2}: Error reading CSV: {e}")
            all_valid = False
            row["_skip"] = True
    
    return all_valid


def run_topic_model(att: str, domain: str, csv_path: str) -> tuple[bool, str]:
    """
    Run the topic model for a single participant/domain combination.
    
    Returns:
        (success: bool, message: str)
    """
    domain_prefix = f"{att}_{domain}"
    cmd = [
        "python",
        "src/kidlink_topics/topic_model.py",
        f"data/{csv_path}",
        "--domain-prefix",
        domain_prefix
    ]
    
    logger = logging.getLogger()
    logger.info(f"Starting: {domain_prefix}")
    logger.debug(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per model
        )
        
        if result.returncode == 0:
            message = f"✓ Completed: {domain_prefix}"
            logger.info(message)
            return True, message
        else:
            message = f"✗ Failed: {domain_prefix}\n  Error: {result.stderr[:1000]}"
            logger.error(message)
            return False, message
            
    except subprocess.TimeoutExpired:
        message = f"✗ Timeout: {domain_prefix} (exceeded 1 hour)"
        logger.error(message)
        return False, message
    except Exception as e:
        message = f"✗ Exception: {domain_prefix}\n  Error: {str(e)}"
        logger.error(message)
        return False, message


def process_batch(config_csv: Path):
    """Process all topic models in batch."""
    logger = logging.getLogger()
    
    # Read configuration
    logger.info(f"Reading configuration from: {config_csv}")
    
    try:
        with open(config_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_csv}")
        return False
    except Exception as e:
        logger.error(f"Error reading configuration file: {e}")
        return False
    
    if not rows:
        logger.error("Configuration file is empty or has no data rows")
        return False
    
    logger.info(f"Found {len(rows)} processing tasks")
    
    # Validate inputs
    logger.info("Validating input files...")
    if not validate_inputs(config_csv, rows):
        logger.warning("Some validation errors, but continuing with valid entries...")
    logger.info("Validation passed ✓")
    
    # Process each row
    results = {"successful": [], "failed": [], "skipped": []}
    start_time = datetime.now()
    
    for idx, row in enumerate(rows, start=1):
        att = row["att"].strip()
        domain = row["domain"].strip()
        csv_name = row["csv-name"].strip()
        domain_prefix = f"{att}_{domain}"
        
        # Skip if marked during validation
        if row.get("_skip", False):
            results["skipped"].append({
                "att": att,
                "domain": domain,
                "csv_name": csv_name,
                "domain_prefix": domain_prefix
            })
            continue
        
        logger.info(f"\n--- Task {idx}/{len(rows)} ---")
        success, message = run_topic_model(att, domain, csv_name)
        
        if success:
            results["successful"].append({
                "att": att,
                "domain": domain,
                "csv_name": csv_name,
                "domain_prefix": domain_prefix
            })
        else:
            results["failed"].append({
                "att": att,
                "domain": domain,
                "csv_name": csv_name,
                "domain_prefix": domain_prefix,
                "error": message
            })
    
    # Summary
    elapsed = datetime.now() - start_time
    logger.info("\n" + "="*60)
    logger.info("BATCH PROCESSING SUMMARY")
    logger.info("="*60)
    logger.info(f"Total tasks: {len(rows)}")
    logger.info(f"Successful: {len(results['successful'])}")
    logger.info(f"Skipped (too small): {len(results['skipped'])}")
    logger.info(f"Failed: {len(results['failed'])}")
    logger.info(f"Elapsed time: {elapsed}")
    
    if results["skipped"]:
        logger.info("\nSkipped tasks (insufficient data):")
        for item in results["skipped"]:
            logger.info(f"  - {item['domain_prefix']}")
    
    if results["failed"]:
        logger.info("\nFailed tasks:")
        for item in results["failed"]:
            logger.info(f"  - {item['domain_prefix']}: {item['error']}")
    
    return len(results["failed"]) == 0


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python batch_topic_models.py <config_csv>")
        print("\nExample:")
        print("  python batch_topic_models.py workshop_participants.csv")
        print("\nConfiguration CSV format (with header):")
        print("  att,domain,csv-name")
        print("  john_doe,healthcare,john_healthcare_data.csv")
        print("  john_doe,education,john_education_data.csv")
        sys.exit(1)
    
    config_csv = Path(sys.argv[1])
    
    # Setup logging
    log_file = setup_logging()
    logger = logging.getLogger()
    
    logger.info("="*60)
    logger.info("BATCH TOPIC MODEL GENERATION")
    logger.info("="*60)
    logger.info(f"Configuration: {config_csv}")
    logger.info(f"Logging to: {log_file}")
    
    # Process batch
    success = process_batch(config_csv)
    
    logger.info("\n" + ("BATCH COMPLETED SUCCESSFULLY ✓" if success else "BATCH COMPLETED WITH ERRORS ✗"))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
