#!/usr/bin/env node
/**
 * Generate deck.pdf from index.html using Puppeteer
 * Usage: node generate-pdf.js
 */

const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

(async () => {
  const htmlPath = path.resolve(__dirname, 'index.html');
  const pdfPath = path.resolve(__dirname, 'deck.pdf');

  console.log('Launching headless browser...');
  const browser = await puppeteer.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  console.log('Loading HTML:', htmlPath);
  const page = await browser.newPage();

  // Load the HTML file
  const htmlContent = fs.readFileSync(htmlPath, 'utf8');
  await page.setContent(htmlContent);

  // Wait for fonts and resources to load
  await page.waitForFunction(() => {
    return document.fonts && document.fonts.ready !== 'loading';
  }, {}, 3000);

  // Generate PDF with print CSS media
  console.log('Generating PDF:', pdfPath);
  await page.pdf({
    path: pdfPath,
    format: 'A3',
    landscape: true,
    printBackground: true,
    preferCSSPageSize: true,
    margin: { top: 0, right: 0, bottom: 0, left: 0 }
  });

  await browser.close();
  console.log('✓ deck.pdf generated successfully');
})();
