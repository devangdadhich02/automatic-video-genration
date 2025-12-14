/**
 * Google Apps Script: scrape content (reviews / SNS / web) and send to backend.
 *
 * Deploy this as a web app or time-based trigger in Google Apps Script.
 * Adjust the `SCRAPE_SOURCE` and scraping logic for your target sites.
 */

const BACKEND_URL = 'https://YOUR_BACKEND_URL/ingest/script'; // e.g. https://example.com/ingest/script
const SCRAPE_SOURCE = 'google-apps-script';

/**
 * Example: scrape rows from a Google Sheet that already contains
 * reviews or SNS text, then send them to the backend RAG endpoint.
 */
function scrapeAndSendToBackend() {
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
  const values = sheet.getDataRange().getValues();

  // Assumes first row is header, text in first column
  let allText = '';
  for (let i = 1; i < values.length; i++) {
    const row = values[i];
    if (row[0]) {
      allText += row[0] + '\n\n';
    }
  }

  if (!allText) {
    Logger.log('No text found to ingest.');
    return;
  }

  const payload = {
    source: SCRAPE_SOURCE,
    text: allText,
    metadata: {
      sheetName: sheet.getName(),
      spreadsheetId: sheet.getParent().getId()
    }
  };

  const options = {
    method: 'post',
    contentType: 'application/json',
    payload: JSON.stringify(payload),
    muteHttpExceptions: true
  };

  const response = UrlFetchApp.fetch(BACKEND_URL, options);
  Logger.log('Backend response: ' + response.getContentText());
}


