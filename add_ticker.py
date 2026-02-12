"""
Simple script to add a ticker to the Supabase watchlist table.

Usage:
    python add_ticker.py
"""

import os
from dotenv import load_dotenv
from supabase import create_client


def main():
    """Prompt user for ticker and add it to Supabase watchlist."""
    # Load environment variables
    load_dotenv()
    
    # Get Supabase credentials
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        print("Error: SUPABASE_URL and SUPABASE_KEY must be set in .env file")
        return
    
    # Create Supabase client
    supabase = create_client(supabase_url, supabase_key)
    
    # Prompt user for ticker
    ticker = input("Enter ticker symbol to add to watchlist: ").strip().upper()
    
    if not ticker:
        print("Error: Ticker cannot be empty")
        return
    
    # Insert ticker into watchlist table
    try:
        response = supabase.table("watchlist").insert({"ticker": ticker}).execute()
        print(f"Successfully added {ticker} to watchlist!")
    except Exception as e:
        print(f"Error adding ticker to watchlist: {e}")


if __name__ == "__main__":
    main()
