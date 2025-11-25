#!/usr/bin/env python3
"""
Setup script for initializing Qdrant and loading sample medical documents.

This script demonstrates how to:
1. Initialize a Qdrant instance
2. Create a collection for medical documents
3. Load sample medical documents
4. Test similarity search
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.qdrant_pipeline import QdrantPipeline
from src.utils.document_processor import DocumentProcessor
from langchain_core.documents import Document
from config_template import (
    GOOGLE_API_KEY,
    QDRANT_URL,
    QDRANT_API_KEY,
    MEDICAL_COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_DIMENSION,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_sample_documents() -> list:
    """Create sample medical documents for demonstration."""
    sample_docs = [
        Document(
            page_content="""
            Hypertension (High Blood Pressure)
            
            Hypertension is a chronic medical condition in which the blood pressure in the arteries is persistently elevated.
            It is one of the most common chronic conditions and a major risk factor for cardiovascular disease.
            
            Classification:
            - Normal: Less than 120/80 mmHg
            - Elevated: Systolic 120-129 and diastolic less than 80
            - Stage 1 Hypertension: Systolic 130-139 or diastolic 80-89
            - Stage 2 Hypertension: Systolic 140 or higher or diastolic 90 or higher
            
            Risk Factors:
            - Age (increases with age)
            - Family history
            - Obesity
            - Sedentary lifestyle
            - High sodium diet
            - Excessive alcohol consumption
            - Stress
            
            Treatment:
            - Lifestyle modifications (diet, exercise, weight loss)
            - Pharmacological treatment with antihypertensive medications
            - Regular monitoring of blood pressure
            """,
            metadata={
                "source": "Medical Knowledge Base - Hypertension",
                "topic": "Cardiovascular Diseases",
                "date": "2024-01-01",
            },
        ),
        Document(
            page_content="""
            Diabetes Mellitus
            
            Diabetes mellitus is a metabolic disorder characterized by hyperglycemia (elevated blood glucose levels)
            resulting from defects in insulin secretion, insulin action, or both.
            
            Types:
            - Type 1 Diabetes: Autoimmune destruction of pancreatic beta cells
            - Type 2 Diabetes: Insulin resistance and relative insulin deficiency
            - Gestational Diabetes: Glucose intolerance during pregnancy
            
            Symptoms:
            - Polyuria (excessive urination)
            - Polydipsia (excessive thirst)
            - Polyphagia (excessive hunger)
            - Fatigue
            - Blurred vision
            - Slow wound healing
            
            Complications:
            - Diabetic retinopathy
            - Diabetic nephropathy
            - Diabetic neuropathy
            - Cardiovascular disease
            - Diabetic ketoacidosis
            
            Management:
            - Blood glucose monitoring
            - Dietary management
            - Physical activity
            - Insulin therapy (for Type 1)
            - Oral antidiabetic medications
            """,
            metadata={
                "source": "Medical Knowledge Base - Diabetes",
                "topic": "Endocrine Disorders",
                "date": "2024-01-01",
            },
        ),
        Document(
            page_content="""
            Pneumonia
            
            Pneumonia is an infection of the lung parenchyma caused by bacteria, viruses, fungi, or parasites.
            It is a leading cause of morbidity and mortality worldwide.
            
            Classification:
            - Community-Acquired Pneumonia (CAP)
            - Hospital-Acquired Pneumonia (HAP)
            - Ventilator-Associated Pneumonia (VAP)
            - Aspiration Pneumonia
            
            Common Causative Organisms:
            - Streptococcus pneumoniae
            - Haemophilus influenzae
            - Mycoplasma pneumoniae
            - Legionella pneumophila
            - Influenza virus
            - COVID-19 virus
            
            Clinical Presentation:
            - Cough (productive or dry)
            - Fever
            - Dyspnea (shortness of breath)
            - Chest pain
            - Fatigue
            
            Diagnosis:
            - Chest X-ray
            - Sputum culture
            - Blood cultures
            - PCR testing (for viral pathogens)
            
            Treatment:
            - Antibiotics (empirical or targeted)
            - Supportive care
            - Oxygen therapy if needed
            - Fluid management
            """,
            metadata={
                "source": "Medical Knowledge Base - Pneumonia",
                "topic": "Respiratory Diseases",
                "date": "2024-01-01",
            },
        ),
        Document(
            page_content="""
            Acute Myocardial Infarction (Heart Attack)
            
            Acute myocardial infarction (AMI) is the death of heart muscle tissue due to inadequate blood supply,
            usually caused by rupture of a coronary artery plaque.
            
            Types:
            - STEMI (ST-Elevation Myocardial Infarction)
            - NSTEMI (Non-ST-Elevation Myocardial Infarction)
            
            Risk Factors:
            - Hypertension
            - Hyperlipidemia
            - Diabetes
            - Smoking
            - Obesity
            - Sedentary lifestyle
            - Family history
            - Male gender
            - Advanced age
            
            Symptoms:
            - Chest pain or pressure
            - Shortness of breath
            - Nausea
            - Sweating
            - Palpitations
            - Syncope
            
            Acute Management:
            - Aspirin administration
            - Antiplatelet therapy
            - Anticoagulation
            - Beta-blockers
            - ACE inhibitors
            - Statins
            - Reperfusion therapy (PCI or thrombolysis)
            
            Complications:
            - Cardiogenic shock
            - Arrhythmias
            - Heart failure
            - Mechanical complications
            - Sudden cardiac death
            """,
            metadata={
                "source": "Medical Knowledge Base - Cardiology",
                "topic": "Cardiovascular Diseases",
                "date": "2024-01-01",
            },
        ),
        Document(
            page_content="""
            Sepsis and Septic Shock
            
            Sepsis is a life-threatening condition that arises when the body's response to infection causes tissue damage.
            Septic shock is sepsis with hypotension despite adequate fluid resuscitation.
            
            Definition (qSOFA Criteria):
            - Altered mental status
            - Systolic blood pressure ≤100 mmHg
            - Respiratory rate ≥22/min
            
            Common Sources of Infection:
            - Pneumonia
            - Urinary tract infection
            - Intra-abdominal infection
            - Meningitis
            - Bacteremia
            
            Pathophysiology:
            - Release of inflammatory mediators
            - Endothelial dysfunction
            - Microvascular thrombosis
            - Organ dysfunction
            
            Clinical Features:
            - Fever or hypothermia
            - Tachycardia
            - Tachypnea
            - Hypotension
            - Altered mental status
            - Oliguria
            
            Management:
            - Early recognition and diagnosis
            - Blood cultures before antibiotics
            - Broad-spectrum antibiotics
            - Fluid resuscitation
            - Vasopressors if needed
            - Source control
            - Supportive care
            - ICU admission
            """,
            metadata={
                "source": "Medical Knowledge Base - Critical Care",
                "topic": "Infectious Diseases",
                "date": "2024-01-01",
            },
        ),
    ]

    return sample_docs


def main():
    """Main setup function."""
    logger.info("Starting Qdrant setup...")

    try:
        # Initialize Qdrant pipeline
        logger.info(f"Initializing Qdrant pipeline at {QDRANT_URL}")
        qdrant_pipeline = QdrantPipeline(
            qdrant_url=QDRANT_URL,
            qdrant_api_key=QDRANT_API_KEY,
            google_api_key=GOOGLE_API_KEY,
            collection_name=MEDICAL_COLLECTION_NAME,
            embedding_dimension=EMBEDDING_DIMENSION,
        )

        # Get collection info
        logger.info("Checking collection status...")
        collection_info = qdrant_pipeline.get_collection_info()
        logger.info(f"Collection info: {collection_info}")

        # Create sample documents
        logger.info("Creating sample medical documents...")
        sample_docs = create_sample_documents()
        logger.info(f"Created {len(sample_docs)} sample documents")

        # Add documents to vector store
        logger.info("Adding documents to vector store...")
        doc_ids = qdrant_pipeline.add_documents(sample_docs)
        logger.info(f"Added {len(doc_ids)} documents successfully")

        # Test similarity search
        logger.info("\n" + "="*60)
        logger.info("Testing similarity search...")
        logger.info("="*60)

        test_queries = [
            "What are the symptoms of hypertension?",
            "How is diabetes diagnosed and treated?",
            "What causes pneumonia?",
            "What is the treatment for heart attack?",
        ]

        for query in test_queries:
            logger.info(f"\nQuery: {query}")
            results = qdrant_pipeline.similarity_search(query, k=2)

            for i, (doc, score) in enumerate(results, 1):
                logger.info(f"  Result {i} (Score: {score:.2f})")
                logger.info(f"  Source: {doc.metadata.get('source', 'Unknown')}")
                logger.info(f"  Content: {doc.page_content[:100]}...")

        # Final status
        logger.info("\n" + "="*60)
        logger.info("Setup completed successfully!")
        logger.info("="*60)

        final_info = qdrant_pipeline.get_collection_info()
        logger.info(f"Final collection status: {final_info}")

    except Exception as e:
        logger.error(f"Error during setup: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
