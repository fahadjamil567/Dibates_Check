# Disease Detection Web Application

A web-based application that uses deep learning to detect diseases in medical images. The application is built with FastAPI for the backend API and a modern HTML/JavaScript frontend.

## Features

- Real-time disease detection from uploaded images
- Modern, responsive UI with Tailwind CSS
- Interactive visualization of prediction results
- Supports various image formats
- Fast and efficient processing using PyTorch

## Tech Stack

- Backend: FastAPI (Python)
- Frontend: HTML, JavaScript, Tailwind CSS
- Machine Learning: PyTorch
- Deployment: Vercel

## Local Development

1. Clone the repository:
```bash
git clone [your-repository-url]
cd disease-detection-app
```

2. Install dependencies:
```bash
pip install -r requirements-vercel.txt
```

3. Run the development server:
```bash
uvicorn api.index:app --reload
```

4. Open `public/index.html` in your browser or serve it using a local server.

## Deployment

The application is configured for deployment on Vercel:

1. Install Vercel CLI:
```bash
npm install -g vercel
```

2. Deploy to Vercel:
```bash
vercel
```

## Project Structure

- `/api` - FastAPI backend code
- `/public` - Static frontend files
- `requirements-vercel.txt` - Python dependencies
- `vercel.json` - Vercel deployment configuration

## License

MIT License 