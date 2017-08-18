//
//  CollectionViewController.swift
//  OpenCV Demo
//
//  Created by Drew Nibeck on 7/31/17.
//  Copyright © 2017 loc. All rights reserved.
//

import UIKit

class CollectionViewController: UIViewController, UICollectionViewDelegate, UICollectionViewDataSource, UICollectionViewDelegateFlowLayout, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
    @IBOutlet weak var collectionView: UICollectionView!
    private var findMeArray = [UIImage]()
    private var imagePicker = UIImagePickerController()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        imagePicker.delegate = self
        
        findMeArray = populateDataSource()
        setupCollectionView()
    }
    
    func populateDataSource() -> [UIImage] {
        var tmp = [UIImage]()
        
        let club  = UIImage(named: "club_query_mask")!
        let mask  = UIImage(named: "mask_query_mask")!
        let piano = UIImage(named: "piano_query_mask")!
        let pot   = UIImage(named: "pot_query_mask")!
        let ram   = UIImage(named: "ram_query_mask")!
        let otherMask = UIImage(named: "mask_train2")!
        let octocat = UIImage(named: "octocat")!
        
        tmp.append(club)
        tmp.append(mask)
        tmp.append(piano)
        tmp.append(pot)
        tmp.append(ram)
        tmp.append(otherMask)
        tmp.append(octocat)
        
        return tmp
    }
    
    func setupCollectionView() {
        (collectionView.collectionViewLayout as! UICollectionViewFlowLayout).sectionInset = UIEdgeInsetsMake(20, 0, 20, 0)
        
        collectionView.allowsMultipleSelection = false
        collectionView.delegate = self
        collectionView.dataSource = self
    }
    
    @IBAction func takeYourOwn(_ sender: Any) {
        let pickerAlert = UIAlertController(title: "From where?", message: "Would you like to take a photo or choose an existing one?", preferredStyle: .actionSheet)
        
        let cameraAlertAction = UIAlertAction(title: "Camera", style: .default) { _ in
            self.imagePicker.sourceType = .camera
            
            self.present(self.imagePicker, animated: true, completion: nil)
        }
        
        let photosAlertAction = UIAlertAction(title: "Photos", style: .default) { _ in
            self.imagePicker.sourceType = .photoLibrary
            
            self.present(self.imagePicker, animated: true, completion: nil)
        }
        
        let cancelAlertAction = UIAlertAction(title: "Cancel", style: .cancel) { _ in
            self.dismiss(animated: true, completion: nil)
        }
        
        pickerAlert.addAction(cameraAlertAction)
        pickerAlert.addAction(photosAlertAction)
        pickerAlert.addAction(cancelAlertAction)
        
        present(pickerAlert, animated: true, completion: nil)
    }
    
    // MARK: -UICollectionViewDelegate
    func collectionView(_ collectionView: UICollectionView, didSelectItemAt indexPath: IndexPath) {
        let detailView = storyboard?.instantiateViewController(withIdentifier: "detailViewController") as! DetailViewController
        detailView.image = findMeArray[indexPath.row]
        
        navigationController?.pushViewController(detailView, animated: true)
    }
    
    // MARK: -UICollectionViewDataSource
    func collectionView(_ collectionView: UICollectionView, cellForItemAt indexPath: IndexPath) -> UICollectionViewCell {
        let localCell = collectionView.dequeueReusableCell(withReuseIdentifier: "scavengerItem", for: indexPath)
        localCell.backgroundView = UIImageView(image: findMeArray[indexPath.row], highlightedImage: nil)
        
        return localCell
    }
    
    func numberOfSections(in collectionView: UICollectionView) -> Int {
        return 1
    }
    
    func collectionView(_ collectionView: UICollectionView, numberOfItemsInSection section: Int) -> Int {
        return findMeArray.count
    }
    
    func collectionView(_ collectionView: UICollectionView, viewForSupplementaryElementOfKind kind: String, at indexPath: IndexPath) -> UICollectionReusableView {
        let headerView = collectionView.dequeueReusableSupplementaryView(ofKind: kind, withReuseIdentifier: "header", for: indexPath)
        
        let header = UITextView(frame: headerView.bounds)
        header.textAlignment = .center
        header.font = UIFont(name: "System", size: 17.0)
        header.text = "Try and find me"
        
        headerView.addSubview(header)
        
        return headerView
    }
    
    // MARK: -UICollectionViewDelegateFlowLayout
    func collectionView(_ collectionView: UICollectionView, layout collectionViewLayout: UICollectionViewLayout, sizeForItemAt indexPath: IndexPath) -> CGSize {
        let collectionWidth = collectionView.frame.size.width
        let cellWidth = collectionWidth / 5.0
        let size = CGSize(width: cellWidth, height: cellWidth);
        
        return size;
    }
    
    // MARK: -UIImagePickerControllerDelegate
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {
        if let pickedImage = info[UIImagePickerControllerOriginalImage] as? UIImage {
            findMeArray.append(pickedImage)
            
            self.collectionView.reloadData()
        }
        
        dismiss(animated: true, completion: nil)
    }
}
