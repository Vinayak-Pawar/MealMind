# MealMind 🍳

An AI-powered Recipe Recommendation System with advanced computer vision and generative AI capabilities.

## 🚀 Features

### Core Features
* **Smart Recipe Recommendations** - AI-powered recipe suggestions based on available ingredients
* **Ingredient Detection** - Real-time camera-based ingredient recognition using computer vision
* **Recipe Video Generation** - AI-generated cooking tutorial videos using Google Flow
* **Recipe Image Generation** - Beautiful recipe images using Nano-Banana AI
* **Nutrition Tracking** - Comprehensive nutritional analysis
* **Dietary Management** - Support for various dietary preferences and restrictions
* **Shopping List Generation** - Automated shopping lists from recipes

### Advanced AI Features
* **Computer Vision** - Object detection for ingredient identification
* **Generative Video** - AI-powered cooking instruction videos
* **Generative Images** - AI-created recipe photography
* **Recommender System** - Machine learning-based recipe suggestions

## 🛠️ Tech Stack

### Backend
- **Python 3.9+** - Core language
- **FastAPI** - Modern web framework
- **SQLAlchemy** - Database ORM
- **OpenCV** - Computer vision for ingredient detection
- **TensorFlow/PyTorch** - Machine learning models
- **Google Flow API** - Video generation
- **Nano-Banana API** - Image generation

### Frontend
- **React.js** - User interface
- **Material-UI** - Component library
- **Axios** - HTTP client

### Database
- **PostgreSQL** - Primary database
- **SQLite** - Development database

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Node.js 16+
- PostgreSQL (optional, SQLite for development)
- Camera access (for ingredient detection)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Vinayak-Pawar/MealMind.git
cd MealMind
```

2. **Backend Setup**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

3. **Frontend Setup**
```bash
# Install Node.js dependencies
npm install
```

4. **Database Setup**
```bash
# For development (SQLite)
python -m alembic upgrade head

# For production (PostgreSQL)
# Update DATABASE_URL in .env file
```

### Usage

1. **Start Backend**
```bash
# Development
uvicorn app.main:app --reload

# Production
python -m gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

2. **Start Frontend**
```bash
npm start
```

3. **Access Application**
- Backend API: http://localhost:8000
- Frontend: http://localhost:3000
- API Documentation: http://localhost:8000/docs

## 📁 Project Structure

```
MealMind/
├── backend/                          # FastAPI Backend
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                  # FastAPI application
│   │   ├── config.py                # Configuration settings
│   │   ├── database.py              # Database connection
│   │   ├── models/                  # SQLAlchemy models
│   │   │   ├── __init__.py
│   │   │   ├── user.py
│   │   │   ├── recipe.py
│   │   │   ├── ingredient.py
│   │   │   └── nutrition.py
│   │   ├── routes/                  # API routes
│   │   │   ├── __init__.py
│   │   │   ├── auth.py
│   │   │   ├── recipes.py
│   │   │   ├── ingredients.py
│   │   │   ├── camera.py
│   │   │   ├── video_generation.py
│   │   │   └── image_generation.py
│   │   ├── services/                # Business logic
│   │   │   ├── __init__.py
│   │   │   ├── recommender.py       # ML recommender system
│   │   │   ├── camera_service.py    # Camera/Computer vision
│   │   │   ├── video_service.py     # Google Flow integration
│   │   │   ├── image_service.py     # Nano-Banana integration
│   │   │   └── nutrition_service.py # Nutrition calculations
│   │   ├── utils/                   # Utilities
│   │   │   ├── __init__.py
│   │   │   ├── image_processing.py
│   │   │   └── ml_utils.py
│   │   └── schemas/                 # Pydantic schemas
│   │       ├── __init__.py
│   │       ├── recipe.py
│   │       └── user.py
│   ├── tests/                       # Unit & integration tests
│   ├── alembic/                     # Database migrations
│   ├── requirements.txt
│   └── .env                         # Environment variables
├── frontend/                         # React Frontend
│   ├── public/
│   ├── src/
│   │   ├── components/              # React components
│   │   │   ├── Camera/
│   │   │   ├── RecipeCard/
│   │   │   ├── IngredientDetector/
│   │   │   └── Navigation/
│   │   ├── pages/                   # Page components
│   │   │   ├── Home/
│   │   │   ├── RecipeDetails/
│   │   │   ├── CameraScan/
│   │   │   └── Profile/
│   │   ├── services/                # API services
│   │   ├── hooks/                   # Custom React hooks
│   │   ├── utils/                   # Frontend utilities
│   │   └── App.js
│   ├── package.json
│   └── .env
├── ml_models/                        # Machine Learning Models
│   ├── ingredient_detection/         # Computer vision models
│   ├── recommender_system/           # Recommendation models
│   └── training_scripts/             # Model training code
├── data/                             # Data files
│   ├── recipes/                      # Recipe datasets
│   ├── ingredients/                  # Ingredient databases
│   └── nutrition/                    # Nutrition information
├── docs/                             # Documentation
│   ├── api/                          # API documentation
│   └── architecture/                 # System architecture docs
├── docker/                           # Docker configurations
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── docker-compose.yml
├── .gitignore
├── README.md
└── LICENSE
```

## License

Apache-2.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

